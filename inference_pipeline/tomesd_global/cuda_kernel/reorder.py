import torch

import reorder_kernel  # compiled module

def load_from_memory(memory: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    assert memory.dim() == 3
    N, _, C = memory.shape
    k = order.shape[0]
    assert memory.is_cuda and order.is_cuda

    output = torch.empty((k, C), device=memory.device, dtype=memory.dtype)

    reorder_kernel.launch(
        memory.contiguous(),
        order.contiguous().to(torch.int32),  # ensure int32 for CUDA
        output,
        k,
        C
    )
    return output

if __name__ == "__main__":
    # memory = torch.randn(4, 4, 2, device='cuda')
    # print(memory)
    # order = torch.tensor([15,14,13,12,11,10,9], device='cuda', dtype=torch.int32)
    # output = load_from_memory(memory, order)
    # print(output)
    # print("Is output contiguous?", output.is_contiguous())

    memory = torch.randn(64, 64, 3072, device='cuda')
    order = torch.arange(1023, 1, -1, device='cuda', dtype=torch.int32)

    # Warm up
    for _ in range(5):
        A = torch.randn(1024, 64, device='cuda')
        B = torch.randn(1024, 64, device='cuda')
        C = A @ B.T

    # Time custom kernel
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch_list = []

    for _ in range(1000):
        start.record()
        output = memory.reshape(-1, memory.size(-1))[order] 
        end.record()
        torch.cuda.synchronize()
        # print("PyTorch indexing time (ms):", start.elapsed_time(end))
        # print("Is output contiguous?", output.is_contiguous())
        torch_list.append(start.elapsed_time(end))
    print("PyTorch indexing median time (ms):", 
          torch.median(torch.tensor(torch_list)).item())
    
    kernel_list = []

    for _ in range(1000):
        start.record()
        kernel_output = load_from_memory(memory, order)
        end.record()
        torch.cuda.synchronize()
        # print("Custom kernel time (ms):", start.elapsed_time(end))
        # print("Is output contiguous?", kernel_output.is_contiguous())
        kernel_list.append(start.elapsed_time(end))
    print("Custom kernel median time (ms):", 
          torch.median(torch.tensor(kernel_list)).item())

    # print(kernel_output)
    print(torch.allclose(output, kernel_output))
