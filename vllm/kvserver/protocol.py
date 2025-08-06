# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
from typing import Union

import msgspec
import torch

from vllm.kvserver.wrapper import CudaIPCWrapper


class KVServerCmd(enum.Enum):
    HANDSHAKE_SCHEDULER = enum.auto()
    HANDSHAKE_WORKER = enum.auto()
    HEARTBEAT = enum.auto()
    OFFLOAD_REQUEST = enum.auto()
    OFFLOAD_FINISHED = enum.auto()
    ONLOAD_REQUEST = enum.auto()
    ONLOAD_FINISHED = enum.auto()


class KVServerMsgBase(msgspec.Struct, tag=True):
    pass


class KVServerHandshakeSchedulerMsg(KVServerMsgBase):
    engine_id: str
    model_name: str

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerHandshakeSchedulerMsg":
        return msgspec.msgpack.decode(payload,
                                      type=KVServerHandshakeSchedulerMsg)


class KVServerHandshakeWorkerMsg(KVServerMsgBase):
    engine_id: str
    model_name: str
    rank: int
    world_size: int
    s_gpu_blocks: list[bytes]

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerHandshakeWorkerMsg":
        return msgspec.msgpack.decode(payload, type=KVServerHandshakeWorkerMsg)


class KVServerOffloadRequest(KVServerMsgBase):
    engine_id: str
    request_id: str
    token_ids: list[int]
    block_ids: tuple[list[int], ...]

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerOffloadRequest":
        return msgspec.msgpack.decode(payload, type=KVServerOffloadRequest)


class KVServerOffloadFinished(KVServerMsgBase):
    engine_id: str
    request_id: str
    success: bool

    @staticmethod
    def from_payload(payload: bytes) -> "KVServerOffloadFinished":
        return msgspec.msgpack.decode(payload, type=KVServerOffloadFinished)


KVServerMsg = Union[KVServerHandshakeSchedulerMsg, KVServerHandshakeWorkerMsg]

## HELPER FUNCTIONS


def decode_payload(cmd: KVServerCmd, payload: bytes) -> KVServerMsgBase:
    match cmd:
        case KVServerCmd.HANDSHAKE_SCHEDULER:
            return KVServerHandshakeSchedulerMsg.from_payload(payload)
        case KVServerCmd.HANDSHAKE_WORKER:
            return KVServerHandshakeWorkerMsg.from_payload(payload)
        case KVServerCmd.OFFLOAD_REQUEST:
            return KVServerOffloadRequest.from_payload(payload)
        case KVServerCmd.OFFLOAD_FINISHED:
            return KVServerOffloadFinished.from_payload(payload)
        case _:
            raise ValueError(f"Unknown command for decoding: {cmd}")


def encode_cmd(cmd: KVServerCmd) -> bytes:
    return cmd.value.to_bytes(1, byteorder='big')


def decode_cmd(b: bytes) -> KVServerCmd:
    return KVServerCmd(int.from_bytes(b, byteorder='big'))


def send_scheduler_handshake(socket):
    msg = KVServerHandshakeSchedulerMsg(
        engine_id="",
        model_name="",
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart(
        [encode_cmd(KVServerCmd.HANDSHAKE_SCHEDULER), payload])


def send_worker_handshake(socket, rank: int, world_size: int,
                          gpu_kv_caches: list[torch.Tensor]):
    # Serialize the GPU blocks as bytes
    s_gpu_blocks = [
        CudaIPCWrapper(tensor).serialize() for tensor in gpu_kv_caches
    ]

    msg = KVServerHandshakeWorkerMsg(
        engine_id="",
        model_name="",
        rank=rank,
        world_size=world_size,
        s_gpu_blocks=s_gpu_blocks,
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart([encode_cmd(KVServerCmd.HANDSHAKE_WORKER), payload])


def send_offload_request(socket, request_id: str, token_ids: list[int],
                         block_ids: tuple[list[int], ...]):
    msg = KVServerOffloadRequest(
        engine_id="",
        request_id=request_id,
        token_ids=token_ids,
        block_ids=block_ids,
    )
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart([encode_cmd(KVServerCmd.OFFLOAD_REQUEST), payload])
