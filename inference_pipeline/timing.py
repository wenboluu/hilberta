import json
import numpy as np

data_list = []
with open('./timing/compute_merge_in_single.jsonl') as f:
    for line in f:
        if line.strip():  # Skip any empty lines
            d = json.loads(line)
            data_list.append(d['elapsed_time_ms'])

print("Mean elapsed time (ms):", np.mean(data_list))
print("Minimum elapsed time (ms):", np.min(data_list))
