from reloc_triton_kernel import attention
import os 
import torch
import math
import time
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
    f.write("group_size,height,width,triton_time_us,text_sdpa_time_us,full_sdpa_time_us,parallel_time_us\n")

    for group_size in group_sizes:
        for height in height_widths:
            width = height  # Assuming square dimensions
            if height == 1024:
                text_length = 768
            else:
                text_length = 1536
            query_key_value_shape = (height // 16) * (width // 16) + text_length

            # Warmup runs
            for _ in range(10):
                query = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                key = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                value = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                text_query = torch.randn(1, 24, text_length, 128, device=device, dtype=torch.float16)
                attn_scale = math.sqrt(128)
                
                # Warmup Triton
                attention(
                    query, key, value, 
                    text_length,
                    query_key_value_shape - text_length,
                    False, attn_scale, group_size, False)
                
                # Warmup text SDPA
                F.scaled_dot_product_attention(text_query, key, value, scale=attn_scale)
                
                # Warmup full SDPA
                F.scaled_dot_product_attention(query, key, value, scale=attn_scale)
                
            # Clean up before timing
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Track results separately
            triton_results = []
            text_sdpa_results = []
            full_sdpa_results = []
            parallel_results = []

            # Create CUDA streams for parallel execution
            triton_stream = torch.cuda.Stream()
            text_sdpa_stream = torch.cuda.Stream()
            
            for _ in range(10):
                # 单独测试Triton
                query = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                key = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                value = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                text_query = torch.randn(1, 24, text_length, 128, device=device, dtype=torch.float16)
                attn_scale = math.sqrt(128)
                
                # 1. 测试单独的Triton attention
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
                
                # 2. 测试单独的针对text部分的SDPA
                text_sdpa_result = Timer(
                    stmt="F.scaled_dot_product_attention(query, key, value, scale=scale)",
                    globals={
                        "F": F,
                        "query": text_query,
                        "key": key,
                        "value": value,
                        "scale": attn_scale,
                    },
                    num_threads=1,
                ).timeit(100)
                text_sdpa_results.append(text_sdpa_result.median * 1e6)
                
                # 3. 测试单独的完整SDPA
                full_sdpa_result = Timer(
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
                full_sdpa_results.append(full_sdpa_result.median * 1e6)
                
                # 4. 测试并行执行Triton和text SDPA
                # 为并行执行准备独立的数据
                triton_query = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                triton_key = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                triton_value = torch.randn(1, 24, query_key_value_shape, 128, device=device, dtype=torch.float16)
                
                text_sdpa_query = torch.randn(1, 24, text_length, 128, device=device, dtype=torch.float16)
                
                torch.cuda.synchronize()  # 确保之前的操作都完成
                
                # 记录开始时间
                start_time = time.time()
                
                # 在Triton stream中执行Triton attention
                with torch.cuda.stream(triton_stream):
                    for _ in range(100):  # 执行100次来匹配Timer
                        attention(
                            triton_query,
                            triton_key,
                            triton_value,
                            text_length,
                            query_key_value_shape - text_length,
                            False,
                            attn_scale,
                            group_size,
                            False)
                
                # 在text SDPA stream中执行text SDPA
                with torch.cuda.stream(text_sdpa_stream):
                    for _ in range(100):  # 执行100次来匹配Timer
                        F.scaled_dot_product_attention(
                            text_sdpa_query, 
                            triton_key, 
                            triton_value, 
                            scale=attn_scale)
                
                # 等待两个流都完成
                torch.cuda.synchronize()
                
                # 计算总时间
                end_time = time.time()
                parallel_time = (end_time - start_time) * 1e6 / 100  # 转换为微秒并计算平均每次的时间
                parallel_results.append(parallel_time)

            # 计算中位数结果
            triton_median = sorted(triton_results)[len(triton_results) // 2]
            text_sdpa_median = sorted(text_sdpa_results)[len(text_sdpa_results) // 2]
            full_sdpa_median = sorted(full_sdpa_results)[len(full_sdpa_results) // 2]
            parallel_median = sorted(parallel_results)[len(parallel_results) // 2]
            
            # 计算顺序执行的总时间（Triton + Text SDPA）
            sequential_time = triton_median + text_sdpa_median

            # 写入文件
            f.write(f"{group_size},{height},{width},{triton_median:.4f},{text_sdpa_median:.4f},{full_sdpa_median:.4f},{parallel_median:.4f}\n")
            
            # 打印结果
            print(f"group_size={group_size}, height={height}, width={width}")
            print(f"  Triton time: {triton_median:.4f}us")
            print(f"  Text SDPA time: {text_sdpa_median:.4f}us")
            print(f"  Full SDPA time: {full_sdpa_median:.4f}us")
            print(f"  Sequential time (Triton+Text SDPA): {sequential_time:.4f}us")
            print(f"  Parallel time: {parallel_median:.4f}us")
            print(f"  Sequential vs Full SDPA speedup: {full_sdpa_median/sequential_time:.2f}x")
            print(f"  Parallel vs Full SDPA speedup: {full_sdpa_median/parallel_median:.2f}x")
            print(f"  Parallel vs Sequential speedup: {sequential_time/parallel_median:.2f}x")
