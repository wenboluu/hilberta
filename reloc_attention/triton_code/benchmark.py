from reloc_triton_kernel import attention
import os 
import torch
import math
from torch.utils.benchmark import Timer
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configurations
group_sizes = [4, 16]
height_widths = [1024, 2048]
device = "cuda:0"

# File to save results
os.makedirs('/home/sz3684/diffusion/reorder_local_attention/triton_version/tomesd_global/triton_code', exist_ok=True)
output_file = '/home/sz3684/diffusion/reorder_local_attention/triton_version/tomesd_global/triton_code/benchmark_results.txt'

# Open the file for writing
with open(output_file, "w") as f:
    f.write("group_size,height,width,triton_time_us,sdpa_time_us\n")

    for group_size in group_sizes:
        for height in height_widths:
            width = height  # Assuming square dimensions
            text_length = 256
            query_key_value_shape = (height // 16) * (width // 16) + text_length

            # Warmup runs
            for _ in range(10):
                query = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                key = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                value = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                attn_scale = math.sqrt(128)
                attention(
                    query,
                    key,
                    value,
                    text_length,
                    query_key_value_shape - text_length,
                    False,
                    attn_scale,
                    group_size,
                    False)
            # Clean up before timing
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Track results separately
            triton_results = []
            sdpa_results = []

            for _ in range(10):
                query = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                key = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                value = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                attn_scale = math.sqrt(128)

                # Time Triton attention
                triton_result = Timer(
                    stmt="flex_attn(query, key, value, text_length, length, False, scale, group_size, False)",
                    globals={
                        "flex_attn": attention,
                        "query": query,
                        "key": key,
                        "value": value,
                        "text_length": text_length,
                        "length": query_key_value_shape - text_length,
                        "scale": attn_scale,
                        "group_size": group_size,
                    },
                    num_threads=1,
                ).timeit(100)
                triton_results.append(triton_result.median * 1e6)

                # Clean up before timing
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Time SDPA
                sdpa_result = Timer(
                    stmt="F.scaled_dot_product_attention(query, key, value, scale=scale)",
                    globals={
                        "F": F,
                        "query": query,
                        "key": key,
                        "value": value,
                        "scale": attn_scale,
                    },
                    num_threads=1,
                ).timeit(100)
                sdpa_results.append(sdpa_result.median * 1e6)

            # Compute median results
            triton_median = sorted(triton_results)[len(triton_results) // 2]
            sdpa_median = sorted(sdpa_results)[len(sdpa_results) // 2]

            # Write to file
            f.write(f"{group_size},{height},{width},{triton_median:.4f},{sdpa_median:.4f}\n")
            print(f"group_size={group_size}, height={height}, width={width}")
            print(f"  Triton time: {triton_median:.4f}us")
            print(f"  SDPA time: {sdpa_median:.4f}us")
            print(f"  Speedup: {sdpa_median/triton_median:.2f}x")
