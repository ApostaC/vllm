# SPDX-License-Identifier: Apache-2.0
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import zmq
from lmcache.experimental.cache_engine import LMCacheEngine
from lmcache.integration.vllm.vllm_adapter import init_lmcache_engine

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.utils import cdiv, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


# FIXME: Use a different way to generate the rpc path in order
# to avoid cross-vllm instance conflict
def determine_shared_pid(role: KVConnectorRole, is_tp: bool):
    # If TP:
    # - scheduler: use pid
    # - worker: use ppid
    # If non-TP:
    # - scheduler: use ppid
    # - worker: use ppid
    # TODO: Change the hacky logic and have a better way
    # to generate the rpc path
    if is_tp:
        return os.getpid(
        ) if role == KVConnectorRole.SCHEDULER else os.getppid()
    else:
        return os.getppid()


def get_zmq_rpc_path_lmcache(role: KVConnectorRole,
                             is_tp: bool = False) -> str:
    shared_pid = determine_shared_pid(role, is_tp)
    base_url = envs.VLLM_RPC_BASE_PATH
    logger.debug("Base URL: %s", base_url)
    return f"ipc://{base_url}/lmcache_rpc_port_{shared_pid}"


# TODO: move this to LMCache so that we can gracefully close it
class LMCacheLookupClient:

    def __init__(self, role: KVConnectorRole, is_tp: bool):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp)
        self.socket = make_zmq_socket(self.ctx,
                                      socket_path,
                                      zmq.REQ,
                                      bind=False)

    def lookup(self, token_ids: torch.Tensor) -> int:
        request = self.encoder.encode(token_ids)
        self.socket.send_multipart(request, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


class LMCacheLookupServer:

    def __init__(self, lmcache_engine: LMCacheEngine, role: KVConnectorRole,
                 is_tp: bool):
        self.decoder = MsgpackDecoder(torch.Tensor)
        self.ctx = zmq.Context()
        socket_path = get_zmq_rpc_path_lmcache(role, is_tp)
        self.socket = make_zmq_socket(self.ctx,
                                      socket_path,
                                      zmq.REP,
                                      bind=True)

        self.lmcache_engine = lmcache_engine
        self.running = True

        def process_request():
            while self.running:
                try:
                    #request = self.socket.recv()
                    frames = self.socket.recv_multipart(copy=False)
                    token_ids = self.decoder.decode(frames)
                    result = self.lmcache_engine.lookup(token_ids)
                    response = result.to_bytes(4, "big")
                    self.socket.send(response)
                except Exception as e:
                    logger.error("Error in LMCache lookup server: %s", e)
                    break
                #continue

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
        # TODO: close the thread!


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    # FIXME: need to check whether the block ids will be changed after
    #        preemption
    allocated_block_ids: list[int]

    # The number of tokens that has been savd
    num_saved_tokens: int = 0

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will 
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.

        """
        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute],
            allocated_block_ids=new_request.block_ids,
            num_saved_tokens=0,
        )

    def update(
        self,
        cached_request: "CachedRequestData",
    ) -> None:
        """Update the request tracker when a running request is 
        scheduled again
        """
        self.token_ids.extend(cached_request.new_token_ids)
        self.allocated_block_ids.extend(cached_request.new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.

        Returns:
            the request metadata if we need to perform load/save 
            operations, None otherwise.

        Side effects:
            This function will update `tracker.num_saved_tokens` if a save
            operation is needed.
        """
        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = cdiv(tracker.num_saved_tokens, lmcache_chunk_size) * \
                lmcache_chunk_size
        skip_save = tracker.num_saved_tokens > 0 and \
                len(tracker.token_ids) < chunk_boundary

        if skip_save and load_spec is None:
            return None

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = len(tracker.token_ids)
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = torch.tensor(tracker.token_ids)
        num_blocks = len(tracker.allocated_block_ids)
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks."
                "Something might be wrong in scheduling logic!")
            logger.error("Num tokens: %d, num blocks: %d, block size: %d",
                         len(token_ids), num_blocks, block_size)

        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = block_offsets.reshape((1, block_size)) + \
                block_ids.reshape((num_blocks, 1)) * block_size

        slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug("Scheduled to load %d tokens for request %s",
                         load_spec.lmcache_cached_tokens, tracker.req_id)
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            save_spec=save_spec,
            load_spec=load_spec,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        is_tp = vllm_config.parallel_config.tensor_parallel_size > 1
        if role == KVConnectorRole.SCHEDULER:
            self.lookup_client = LMCacheLookupClient(role, is_tp)
        else:
            self.lmcache_engine = init_lmcache_engine(
                vllm_config.model_config, vllm_config.parallel_config,
                vllm_config.cache_config)
            # NOTE: Only create the KV lookup API server on worker rank 0
            # when there are multiple workers
            if vllm_config.parallel_config.rank == 0:
                self.lookup_server = LMCacheLookupServer(
                    self.lmcache_engine, role, is_tp)

        self.kv_caches: dict[str, torch.Tensor] = {}

        self._block_size = vllm_config.cache_config.block_size

        # request_id -> (vllm cached tokes, lmcache cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}

        self.kv_cache_manager: Optional[KVCacheManager] = None

        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}

        # TODO: need to align this chunk size with lmcache
        self._lmcache_chunk_size = 256

    def _init_kv_caches_from_forward_context(
            self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it",
                             layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[\
                        forward_context.virtual_engine]

    ####################
    # Worker side APIs
    ####################

    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's 
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
        """
        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the attn_metadata is None")
            return

        # HACK: getting chunk size to correctly calculate retrieve mask
        lmcache_chunk_size = self.lmcache_engine.config.chunk_size

        for request in metadata.requests:
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = request.slot_mapping.cuda()
            assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones_like(tokens, dtype=torch.bool)
            masked_token_count = request.load_spec.vllm_cached_tokens // \
                lmcache_chunk_size * lmcache_chunk_size
            token_mask[:masked_token_count] = False

            ret_token_mask = self.lmcache_engine.retrieve(
                tokens,
                token_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping)

            # Check the result
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = request.load_spec.lmcache_cached_tokens - \
                    request.load_spec.vllm_cached_tokens
            if num_retrieved_tokens < num_expected_tokens:
                logger.error(
                    "The number of retrieved tokens is less than the "
                    "expected number of tokens! This should not happen!")
                logger.error(
                    "Num retrieved tokens: %d, num expected tokens: %d",
                    num_retrieved_tokens, num_expected_tokens)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. 
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        if layer_name not in self.kv_caches:
            self.kv_caches[layer_name] = kv_layer

    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""
        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        # HACK: getting chunk size to correctly calculate store mask
        lmcache_chunk_size = self.lmcache_engine.config.chunk_size

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue

            token_ids = request.token_ids
            assert isinstance(token_ids, torch.Tensor)
            assert token_ids.is_cpu

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.cuda()
            # NOTE: In PD setting, lmcache_engine.lookup() will always return
            # 0 if there is no local storage configured. In this case, we
            # should rely on the slip_leading_tokens in save_spec to avoid
            # transmit the already saved tokens again.
            skip_leading_tokens = max(self.lmcache_engine.lookup(token_ids),
                                      save_spec.skip_leading_tokens)
            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = skip_leading_tokens // \
                    lmcache_chunk_size * lmcache_chunk_size

            store_mask = torch.ones_like(token_ids, dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for %d out of %d tokens for request %s",
                len(token_ids) - skip_leading_tokens, len(token_ids),
                request.req_id)
            self.lmcache_engine.store(token_ids,
                                      mask=store_mask,
                                      kvcaches=kvcaches,
                                      slot_mapping=slot_mapping,
                                      offset=skip_leading_tokens)

    ###################
    # Scheduler side APIs
    ####################

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        """
        Check for external KV cache hit.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the 
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_producer":
            return 0

        token_ids = torch.tensor(request.prompt_token_ids)
        num_external_hit_tokens = self.lookup_client.lookup(token_ids)

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1

        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: %s, Total tokens %d, LMCache hit tokens: %d, "
            "need to load: %d", request.request_id, request.num_tokens,
            num_external_hit_tokens, need_to_allocate)

        if need_to_allocate <= 0:
            return 0

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False)

        # TODO: Align to vLLM block size. Should test whether it can be removed
        #need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size
        return need_to_allocate

    def update_state_after_alloc(self, request: "Request",
                                 num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        assert num_external_tokens == \
            self.load_specs[request.request_id].lmcache_cached_tokens - \
            self.load_specs[request.request_id].vllm_cached_tokens, \
            f"Mismatch in number of tokens: {num_external_tokens} vs " \
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} - " \
            f"{self.load_specs[request.request_id].vllm_cached_tokens}" \
            f" for request {request.request_id}"

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        self.load_specs[request.request_id].can_load = True

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output 
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)

        for request in scheduler_output.scheduled_new_reqs:
            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = request.num_computed_tokens + \
                    scheduler_output.num_scheduled_tokens[request.req_id]
            request_tracker = RequestTracker.from_new_request(
                request, num_tokens_to_compute)
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(request_tracker,
                                                    self._block_size,
                                                    self._lmcache_chunk_size,
                                                    load_spec)
            if req_meta is not None:
                meta.add_request(req_meta)

        for request in scheduler_output.scheduled_cached_reqs:
            request_tracker = self._request_trackers[request.req_id]
            request_tracker.update(request)

            req_meta = ReqMeta.from_request_tracker(request_tracker,
                                                    self._block_size,
                                                    self._lmcache_chunk_size,
                                                    None)
            if req_meta is not None:
                meta.add_request(req_meta)

        return meta
