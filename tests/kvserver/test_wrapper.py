# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.kvserver.wrapper import (CudaIPCWrapper, decode_cuda_ipc_wrapper,
                                   encode_cuda_ipc_wrapper)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ipc_wrapper_serialization_deserialization():
    """Test CUDA IPC wrapper serialization and deserialization"""
    # Create a test tensor on CUDA
    tensor = torch.randn(1024, 1024).cuda()

    # Create wrapper
    wrapper = CudaIPCWrapper(tensor)

    # Test encoding/decoding
    data = encode_cuda_ipc_wrapper(wrapper)
    restored_wrapper = decode_cuda_ipc_wrapper(data)

    # Verify the restored wrapper matches the original
    assert wrapper == restored_wrapper, \
            "Restored wrapper does not match original"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ipc_wrapper_properties():
    """Test CudaIPCWrapper properties"""
    # Create test tensor with specific properties
    original_tensor = torch.randn(512, 256, dtype=torch.float32).cuda()

    # Create wrapper
    wrapper = CudaIPCWrapper(original_tensor)

    # Verify wrapper properties
    assert wrapper.shape == original_tensor.shape
    assert wrapper.dtype == original_tensor.dtype
    assert wrapper.device == original_tensor.device.index


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ipc_wrapper_equality():
    """Test CudaIPCWrapper equality comparison"""
    # Create identical tensors
    tensor1 = torch.randn(100, 100).cuda()
    tensor2 = tensor1.clone()

    # Create wrappers
    wrapper1 = CudaIPCWrapper(tensor1)
    wrapper2 = CudaIPCWrapper(tensor2)

    # Different tensors should have different wrappers
    assert wrapper1 != wrapper2

    # Same wrapper should equal itself
    assert wrapper1 == wrapper1

    # Serialized and deserialized wrapper should be equal
    data = wrapper1.serialize()
    restored_wrapper = CudaIPCWrapper.deserialize(data)
    assert wrapper1 == restored_wrapper


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ipc_wrapper_different_dtypes():
    """Test CudaIPCWrapper with different tensor dtypes"""
    dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]

    for dtype in dtypes:
        tensor = torch.randn(64, 64).to(dtype=dtype).cuda()
        wrapper = CudaIPCWrapper(tensor)

        # Test serialization/deserialization
        data = encode_cuda_ipc_wrapper(wrapper)
        restored_wrapper = decode_cuda_ipc_wrapper(data)

        assert wrapper == restored_wrapper
        assert wrapper.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_ipc_wrapper_different_shapes():
    """Test CudaIPCWrapper with different tensor shapes"""
    shapes = [(10, ), (10, 20), (5, 10, 15), (2, 3, 4, 5)]

    for shape in shapes:
        tensor = torch.randn(*shape).cuda()
        wrapper = CudaIPCWrapper(tensor)

        # Test serialization/deserialization
        data = encode_cuda_ipc_wrapper(wrapper)
        restored_wrapper = decode_cuda_ipc_wrapper(data)

        assert wrapper == restored_wrapper
        assert wrapper.shape == shape


if __name__ == "__main__":
    # Allow running the test file directly with pytest
    pytest.main([__file__])
