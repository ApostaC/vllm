# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import msgspec
import zmq

from vllm.kvserver.protocol import (KVServerCmd, KVServerHandshakeSchedulerMsg,
                                    KVServerHandshakeWorkerMsg,
                                    KVServerOffloadFinished,
                                    KVServerOffloadRequest, decode_cmd,
                                    decode_payload, encode_cmd)
from vllm.kvserver.wrapper import CudaIPCWrapper
from vllm.utils import make_zmq_path, make_zmq_socket


@dataclass
class KVServerConfig:
    # The host to bind the server to
    host: str
    # The port to bind the protocol socket to
    port: int


ClientId = bytes
RequestId = str
"""
The server module will have a zmq router socket doing the following thing:
    - Listening for init message and heartbeats from vLLMs
    - Receive offload/onload requests from the alive vLLMs
    - Send back the offload/onload status to the alive vLLMs

The main loop will be:
    - Process the incoming requests from clients
      - Immediately process the init message
      - Update the alive status
      - Put the offload/onload requests into a queue
    - Initiate offload/onload jobs in the queue
    - Check the offload/onload job status
    - Send back the offload/onload status to the clients
"""


class KVServer:

    def __init__(self, config: KVServerConfig):
        self.config = config
        self.context = zmq.Context()

        # Protocol socket
        self.zmq_path = make_zmq_path("tcp", config.host, config.port)
        self.main_socket = make_zmq_socket(self.context,
                                           self.zmq_path,
                                           zmq.ROUTER,
                                           bind=True)

        self.poller = zmq.Poller()
        self.poller.register(self.main_socket, zmq.POLLIN)

        self.debug_offload_queue: list[tuple[ClientId, RequestId]] = []

    def debug_process_offload_requests(self):
        # TODO: send the offload response back to the clients
        for client_id, req_id in self.debug_offload_queue:
            print(f"Processing offload request for client "
                  f"{client_id}, request {req_id}")
            # Simulate sending back an offload finished message
            response_msg = KVServerOffloadFinished(engine_id="",
                                                   request_id=req_id,
                                                   success=True)
            response_payload = msgspec.msgpack.encode(response_msg)
            self.main_socket.send_multipart([
                client_id,
                encode_cmd(KVServerCmd.OFFLOAD_FINISHED), response_payload
            ])
        self.debug_offload_queue.clear()

    def process_tasks(self):
        self.debug_process_offload_requests()

    def handle_handshake_scheduler(self, client_id, cmd, payload):
        # Deserialize the handshake message
        msg = decode_payload(cmd, payload)
        assert isinstance(msg, KVServerHandshakeSchedulerMsg)
        print("Got handshake from scheduler")

    def handle_handshake_worker(self, client_id, cmd, payload):
        # Deserialize the worker handshake message
        msg = decode_payload(cmd, payload)
        assert isinstance(msg, KVServerHandshakeWorkerMsg)
        gpu_blocks = [CudaIPCWrapper.deserialize(b) for b in msg.s_gpu_blocks]
        print(f"Got handshake from worker {msg.rank}/"
              f"{msg.world_size} for engine {msg.engine_id}")
        # Print out gpu block info
        for i, block in enumerate(gpu_blocks):
            tensor = block.to_tensor()
            print(f"  GPU Block {i}: shape={tensor.shape}, "
                  f"dtype={tensor.dtype}, device={tensor.device}")

        # TODO: HERE, save the pointer to the gpu blocks

    def handle_heartbeat(self, client_id, cmd, payload):
        # Update the alive status of the client
        print("Received heartbeat from:", client_id)

    def handle_offload_request(self, client_id, cmd, payload):
        msg = decode_payload(cmd, payload)
        assert isinstance(msg, KVServerOffloadRequest)
        print(f"Received offload request from {client_id} for engine "
              f"{msg.engine_id}, request_id {msg.request_id}, blocks "
              f"{msg.block_ids}")
        self.debug_offload_queue.append((client_id, msg.request_id))

    def handle_onload_request(self, client_id, cmd, payload):
        print("Received onload request from:", client_id)

    def step(self):
        # Poll the main socket for incoming messages
        socks = dict(self.poller.poll(timeout=100))

        if self.main_socket in socks and socks[self.main_socket] == zmq.POLLIN:
            # Receive a message
            msg = self.main_socket.recv_multipart()
            client_id = msg[0]
            cmd = decode_cmd(msg[1])
            payload = msg[2]

            if cmd == KVServerCmd.HANDSHAKE_SCHEDULER:
                self.handle_handshake_scheduler(client_id, cmd, payload)
            elif cmd == KVServerCmd.HANDSHAKE_WORKER:
                self.handle_handshake_worker(client_id, cmd, payload)
            elif cmd == KVServerCmd.HEARTBEAT:
                self.handle_heartbeat(client_id, cmd, payload)
            elif cmd == KVServerCmd.OFFLOAD_REQUEST:
                self.handle_offload_request(client_id, cmd, payload)
            elif cmd == KVServerCmd.ONLOAD_REQUEST:
                self.handle_onload_request(client_id, cmd, payload)
            else:
                print(f"Unknown command from client {client_id}: {cmd}")

        self.process_tasks()


if __name__ == "__main__":
    config = KVServerConfig(host="localhost", port=54332)
    server = KVServer(config)
    print("Starting the server at", config.host, ":", config.port)
    while True:
        server.step()
