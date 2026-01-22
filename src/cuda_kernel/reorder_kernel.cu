#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel to load C-dimensional vectors from memory using flattened indices
// memory: [N * N, C] - flattened from original [N, N, C]
// order:  [k]        - flattened indices to be loaded
// output: [k, C]     - output tensor to store selected vectors
__global__ void load_from_memory_kernel(
    const float* __restrict__ memory,   // [N*N, C]
    const int* __restrict__ order,      // [k]
    float* __restrict__ output,         // [k, C]
    int k,
    int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // which element
    int c_start = threadIdx.y;                        // channel offset
    int c_stride = blockDim.y;                        // total threads in y-dim

    if (idx >= k) return;

    int flat_index = order[idx];  // flattened index from order

    for (int c = c_start; c < C; c += c_stride) {
        int mem_idx = flat_index * C + c;
        int out_idx = idx * C + c;
        output[out_idx] = memory[mem_idx];
    }
}


void launch(
    at::Tensor memory,
    at::Tensor order,
    at::Tensor output,
    int k,
    int C
) {
    auto memory_flat = memory.view({-1, C});

    dim3 block(32, 32);                   
    dim3 grid((k + block.x - 1) / block.x);

    load_from_memory_kernel<<<grid, block>>>(
        memory_flat.data_ptr<float>(),
        order.data_ptr<int>(),
        output.data_ptr<float>(),
        k,
        C
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA kernel launch failed");
    }
}

// Register with PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &launch, "CUDA memory reorder kernel with dynamic k");
}
