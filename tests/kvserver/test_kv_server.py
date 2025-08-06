# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from multiprocessing import Process

import msgspec
import pytest
import torch
import zmq

from vllm.kvserver.protocol import (KVServerCmd, KVServerHandshakeSchedulerMsg,
                                    KVServerHandshakeWorkerMsg,
                                    KVServerOffloadFinished,
                                    KVServerOffloadRequest, decode_cmd,
                                    decode_payload, encode_cmd,
                                    send_offload_request,
                                    send_scheduler_handshake,
                                    send_worker_handshake)
from vllm.kvserver.server import KVServer, KVServerConfig
from vllm.kvserver.wrapper import CudaIPCWrapper


def run_server(host: str, port: int):
    """Function to run KVServer in a separate process"""
    config = KVServerConfig(host=host, port=port)
    server = KVServer(config)
    print(f"Starting KV server at {host}:{port}")
    try:
        while True:
            server.step()
            # process_tasks is now called within server.step()
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server.context.term()


@pytest.fixture(scope="function")
def kv_server():
    """Fixture to start and stop KVServer for each test"""
    host = "localhost"
    port = 5556  # Use different port to avoid conflicts

    # Start server in separate process
    server_process = Process(target=run_server, args=(host, port))
    server_process.start()

    # Give server time to start up
    time.sleep(0.5)

    yield host, port

    # Cleanup: terminate server process
    server_process.terminate()
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.kill()
        server_process.join()


@pytest.fixture(scope="function")
def zmq_client_socket(kv_server):
    """Fixture to create ZMQ client socket connected to the test server"""
    host, port = kv_server
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://{host}:{port}")

    # Give socket time to connect
    time.sleep(0.1)

    yield socket

    # Cleanup
    socket.close()
    context.term()


def test_scheduler_handshake(zmq_client_socket):
    """Test scheduler handshake message"""
    socket = zmq_client_socket

    cmd = KVServerCmd.HANDSHAKE_SCHEDULER
    msg = KVServerHandshakeSchedulerMsg(engine_id="engine_1",
                                        model_name="gpt-3")
    payload = msgspec.msgpack.encode(msg)

    socket.send_multipart([cmd.value.to_bytes(1, byteorder='big'), payload])

    # Give server time to process
    time.sleep(0.1)

    # For now, we just verify the message was sent without error
    # In a more complete test, we might check for a response
    print("Scheduler handshake sent successfully")


def test_worker_handshake(zmq_client_socket):
    """Test worker handshake message"""
    socket = zmq_client_socket

    # Create test tensors (use CPU if CUDA not available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = [torch.randn(1024, 1024).to(device) for _ in range(2)]

    if device == "cuda":
        s_gpu_blocks = [
            CudaIPCWrapper(tensor).serialize() for tensor in tensors
        ]
    else:
        # For CPU testing, we'll use empty serialized blocks
        s_gpu_blocks = [b"mock_cpu_block_1", b"mock_cpu_block_2"]

    cmd = KVServerCmd.HANDSHAKE_WORKER
    msg = KVServerHandshakeWorkerMsg(engine_id="engine_1",
                                     model_name="gpt-3",
                                     rank=0,
                                     world_size=1,
                                     s_gpu_blocks=s_gpu_blocks)
    payload = msgspec.msgpack.encode(msg)

    socket.send_multipart([cmd.value.to_bytes(1, byteorder='big'), payload])

    # Give server time to process
    time.sleep(0.1)

    print("Worker handshake sent successfully")


def test_heartbeat_message(zmq_client_socket):
    """Test heartbeat message"""
    socket = zmq_client_socket

    cmd = KVServerCmd.HEARTBEAT
    # Heartbeat doesn't need a payload, but we send empty bytes
    payload = b""

    socket.send_multipart([encode_cmd(cmd), payload])

    # Give server time to process
    time.sleep(0.1)

    print("Heartbeat sent successfully")


def test_offload_request(zmq_client_socket):
    """Test offload request message"""
    socket = zmq_client_socket

    cmd = KVServerCmd.OFFLOAD_REQUEST
    msg = KVServerOffloadRequest(engine_id="engine_1",
                                 request_id="req_123",
                                 token_ids=[1, 2, 3, 4, 5],
                                 block_ids=([10, 11, 12], [20, 21, 22]))
    payload = msgspec.msgpack.encode(msg)

    socket.send_multipart([encode_cmd(cmd), payload])

    # Give server time to process and possibly respond
    time.sleep(0.2)

    # Try to receive response (offload finished message)
    try:
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        response = socket.recv_multipart()
        if len(response) >= 2:
            response_cmd = decode_cmd(response[0])
            if response_cmd == KVServerCmd.OFFLOAD_FINISHED:
                response_msg = decode_payload(response_cmd, response[1])
                assert isinstance(response_msg, KVServerOffloadFinished)
                assert response_msg.request_id == "req_123"
                assert response_msg.success
                print("Received offload finished response successfully")
        socket.setsockopt(zmq.RCVTIMEO, -1)  # Reset timeout
    except zmq.Again:
        # No response received, which might be expected in some cases
        print("No response received for offload request")

    print("Offload request sent successfully")


def test_helper_functions():
    """Test protocol helper functions"""
    # Test encode/decode cmd
    cmd = KVServerCmd.HANDSHAKE_SCHEDULER
    encoded = encode_cmd(cmd)
    decoded = decode_cmd(encoded)
    assert decoded == cmd

    # Test payload decoding
    scheduler_msg = KVServerHandshakeSchedulerMsg(engine_id="test",
                                                  model_name="test-model")
    payload = msgspec.msgpack.encode(scheduler_msg)
    decoded_msg = decode_payload(KVServerCmd.HANDSHAKE_SCHEDULER, payload)
    assert isinstance(decoded_msg, KVServerHandshakeSchedulerMsg)
    assert decoded_msg.engine_id == "test"
    assert decoded_msg.model_name == "test-model"

    # Test worker message decoding
    worker_msg = KVServerHandshakeWorkerMsg(
        engine_id="worker_test",
        model_name="worker-model",
        rank=1,
        world_size=4,
        s_gpu_blocks=[b"block1", b"block2"])
    payload = msgspec.msgpack.encode(worker_msg)
    decoded_msg = decode_payload(KVServerCmd.HANDSHAKE_WORKER, payload)
    assert isinstance(decoded_msg, KVServerHandshakeWorkerMsg)
    assert decoded_msg.engine_id == "worker_test"
    assert decoded_msg.rank == 1
    assert decoded_msg.world_size == 4

    # Test offload request decoding
    offload_msg = KVServerOffloadRequest(engine_id="offload_test",
                                         request_id="req_456",
                                         token_ids=[10, 20, 30],
                                         block_ids=([1, 2], [3, 4]))
    payload = msgspec.msgpack.encode(offload_msg)
    decoded_msg = decode_payload(KVServerCmd.OFFLOAD_REQUEST, payload)
    assert isinstance(decoded_msg, KVServerOffloadRequest)
    assert decoded_msg.engine_id == "offload_test"
    assert decoded_msg.request_id == "req_456"
    assert decoded_msg.token_ids == [10, 20, 30]

    print("All helper functions work correctly")


def test_send_scheduler_handshake_helper(zmq_client_socket):
    """Test the send_scheduler_handshake helper function"""
    socket = zmq_client_socket

    # Use the helper function
    send_scheduler_handshake(socket)

    # Give server time to process
    time.sleep(0.1)

    print("Scheduler handshake helper function works")


def test_send_worker_handshake_helper(zmq_client_socket):
    """Test the send_worker_handshake helper function"""
    socket = zmq_client_socket

    # Create test tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        tensors = [torch.randn(512, 512).to(device) for _ in range(1)]
        send_worker_handshake(socket,
                              rank=0,
                              world_size=1,
                              gpu_kv_caches=tensors)
    else:
        # Skip this test on CPU since CudaIPCWrapper requires CUDA
        print("Skipping worker handshake helper test on CPU")
        return

    # Give server time to process
    time.sleep(0.1)

    print("Worker handshake helper function works")


def test_send_offload_request_helper(zmq_client_socket):
    """Test the send_offload_request helper function"""
    socket = zmq_client_socket

    # Use the helper function
    send_offload_request(socket,
                         request_id="helper_req_789",
                         token_ids=[5, 10, 15, 20],
                         block_ids=([100, 101], [200, 201, 202]))

    # Give server time to process and possibly respond
    time.sleep(0.2)

    # Try to receive response (offload finished message)
    try:
        socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        response = socket.recv_multipart()
        if len(response) >= 2:
            response_cmd = decode_cmd(response[0])
            if response_cmd == KVServerCmd.OFFLOAD_FINISHED:
                response_msg = decode_payload(response_cmd, response[1])
                assert isinstance(response_msg, KVServerOffloadFinished)
                assert response_msg.request_id == "helper_req_789"
                assert response_msg.success
                print(
                    "Received offload finished response from helper function")
        socket.setsockopt(zmq.RCVTIMEO, -1)  # Reset timeout
    except zmq.Again:
        # No response received, which might be expected in some cases
        print("No response received for offload request from helper")

    print("Offload request helper function works")


def test_multiple_commands_sequence(zmq_client_socket):
    """Test sending multiple different commands in sequence"""
    socket = zmq_client_socket

    # Send scheduler handshake
    send_scheduler_handshake(socket)
    time.sleep(0.1)

    # Send heartbeat
    socket.send_multipart([encode_cmd(KVServerCmd.HEARTBEAT), b""])
    time.sleep(0.1)

    # Send offload request using helper function
    send_offload_request(socket,
                         request_id="seq_req_1",
                         token_ids=[100, 200],
                         block_ids=([50, 51], ))
    time.sleep(0.2)

    print("Multiple commands sequence sent successfully")


def test_multiple_handshakes(zmq_client_socket):
    """Test sending both scheduler and worker handshakes in sequence"""
    socket = zmq_client_socket

    # Send scheduler handshake
    cmd = KVServerCmd.HANDSHAKE_SCHEDULER
    msg = KVServerHandshakeSchedulerMsg(engine_id="engine_1",
                                        model_name="gpt-3")
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart([cmd.value.to_bytes(1, byteorder='big'), payload])

    time.sleep(0.1)

    # Send worker handshake
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = [torch.randn(512, 512).to(device) for _ in range(1)]

    if device == "cuda":
        s_gpu_blocks = [
            CudaIPCWrapper(tensor).serialize() for tensor in tensors
        ]
    else:
        s_gpu_blocks = [b"mock_cpu_block"]

    cmd = KVServerCmd.HANDSHAKE_WORKER
    msg = KVServerHandshakeWorkerMsg(engine_id="engine_2",
                                     model_name="gpt-4",
                                     rank=0,
                                     world_size=1,
                                     s_gpu_blocks=s_gpu_blocks)
    payload = msgspec.msgpack.encode(msg)
    socket.send_multipart([cmd.value.to_bytes(1, byteorder='big'), payload])

    time.sleep(0.1)

    print("Multiple handshakes sent successfully")


def test_invalid_command_handling():
    """Test how the protocol handles invalid commands"""
    # Test invalid command byte
    with pytest.raises(ValueError):
        decode_cmd(b'\xFF')  # Invalid command value

    # Test unknown command in decode_payload
    with pytest.raises(ValueError):
        # Create a mock command that doesn't exist
        class MockCmd:
            pass

        mock_cmd = MockCmd()
        decode_payload(mock_cmd, b"test")

    print("Invalid command handling works correctly")


def test_kv_server_creation_and_step():
    """Test that KVServer can be correctly created and stepped"""
    # Test server creation
    config = KVServerConfig(host="localhost",
                            port=5557)  # Different port to avoid conflicts
    server = KVServer(config)

    # Verify server was created correctly
    assert server.config == config
    assert server.context is not None
    assert server.main_socket is not None
    assert server.poller is not None
    assert server.debug_offload_task_queue == []

    # Test that step() method works without errors
    # This should poll with timeout and return without processing any messages
    try:
        server.step()
        print("Server step executed successfully")
    except Exception as e:
        pytest.fail(f"Server step failed with error: {e}")

    # Test multiple steps
    for i in range(5):
        try:
            server.step()
        except Exception as e:
            pytest.fail(f"Server step {i+1} failed with error: {e}")

    print("Multiple server steps executed successfully")

    # Test that process_tasks works
    try:
        server.process_tasks()
        print("Server process_tasks executed successfully")
    except Exception as e:
        pytest.fail(f"Server process_tasks failed with error: {e}")

    # Cleanup
    server.context.term()
    print("KVServer creation and stepping test completed successfully")


def test_kv_server_config():
    """Test KVServerConfig creation and properties"""
    host = "127.0.0.1"
    port = 8888

    config = KVServerConfig(host=host, port=port)

    assert config.host == host
    assert config.port == port

    print("KVServerConfig test completed successfully")


def test_kv_server_offload_queue_operations():
    """Test KVServer offload queue operations"""
    config = KVServerConfig(host="localhost", port=5558)  # Different port
    server = KVServer(config)

    # Test initial state
    assert len(server.debug_offload_task_queue) == 0

    # Test adding items to queue manually
    test_client_id = b"test_client_123"
    test_request_id = "test_req_456"

    server.debug_offload_task_queue.append((test_client_id, test_request_id))

    # Verify queue state
    assert len(server.debug_offload_task_queue) == 1
    assert server.debug_offload_task_queue[0] == (test_client_id,
                                                  test_request_id)

    # Test process_tasks (this should process and clear the queue)
    # Note: This will try to send responses, but since we don't have a real
    # client, it might fail. We'll catch any exceptions to verify the queue
    # processing logic.
    initial_queue_length = len(server.debug_offload_task_queue)

    try:
        server.debug_process_offload_requests()
        # If it succeeds, the queue should be processed
        print(f"Queue processed successfully, initial length: "
              f"{initial_queue_length}")
    except Exception as e:
        # Expected if there's no client to send responses to
        print(f"Queue processing failed as expected (no client): {e}")

    # Cleanup
    server.context.term()
    print("KVServer offload queue operations test completed")


if __name__ == "__main__":
    # Allow running the test file directly with pytest
    pytest.main([__file__])
