# SPDX-License-Identifier: Apache-2.0

"""
KVConnectorBase Class for Distributed KV Cache & Hidden State communication in vLLM v1

The class provides the following primitives:
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

import torch
import enum

from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request
    from vllm.config import VllmConfig

class KVConnectorRole(enum.Enum):
    # Connector running in the scheduler process
    SCHEDULER = 0

    # Connector running in the worker process
    WORKER = 1


class KVConnectorBase(ABC):
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
        role: KVConnectorRole):
        self._connector_metada = None
        self._rank = rank
        self._local_rank = local_rank
        self._config = config
        self._role = role

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    def bind_connector_metadata(
            self, 
            connector_metadata: "KVConnectorMetadata") -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time 
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self._connector_metada = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time 
        after the model execution.
        """
        self._connector_metada = None

    def _get_connector_metadata(self) -> "KVConnectorMetadata":
        """Get the connector metadata.

        This function should only be called inside the connector.

        Returns:
            ConnectorMetadata: the connector metadata.
        """
        return self._connector_metada

    # ==============================
    # Worker-side methods
    # ==============================

    @abstractmethod
    def start_load_kv(
            self, 
            forward_context: "ForwardContext",
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
        pass

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer. 
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        pass

    @abstractmethod
    def save_kv_layer(
            self, 
            layer_name: str, 
            kv_layer: torch.Tensor, 
            attn_metadata: "AttentionMetadata",
            **kwargs) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        pass

    @abstractmethod
    def wait_for_save(self):
        """Block until all the save operations is done. 

        This prevents vLLM overwrites the paged KV buffer before 
        saving is done.
        """
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================
    @abstractmethod
    def get_external_prefix_cache_blocks(
            self, 
            request: "Request",
            computed_blocks: list["KVCacheBlock"],
            num_computed_tokens: int,
            kv_cache_manager: "KVCacheManager",
        ) -> list["KVCacheBlock"]:
        """Get the external prefix cache blocks from the connector.

        This function may change the state of the connector, which will be 
        used by `attach_connector_meta` later.

        This function will also allocate/free the blocks dynamically when  
        there is remote cache hit.

        Args:
            request (Request): the request object.
            computed_blocks (list[KVCacheBlock]): the 'local' computed blocks.
            num_computed_tokens (int): the number of 'local' computed tokens.
            kv_cache_manager (KVCacheManager): the KV cache manager to 
                allocate/free the blocks if needed.

        Returns:
            The updated list of the computed blocks (appended with the remote
            cached blocks)
        """
        pass

    @abstractmethod
    def attach_connector_meta(
            self, 
            scheduler_output: SchedulerOutput
        ) -> SchedulerOutput:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output 
        except the `connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        pass

