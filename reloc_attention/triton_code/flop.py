seq_len_list = [4096, 16384]
shared_len_list = [768, 1536]
num_tile_list = [4, 16]

for i in range(2):
    current_seq_len = seq_len_list[i]
    current_shared_len = shared_len_list[i]
    for j in range(2):
        current_num_tile = num_tile_list[j]
        flop_local_attn = 4 * 24 * (current_seq_len // current_num_tile) * (current_seq_len // current_num_tile + current_shared_len) * 128 * current_num_tile
        flop_shared_attn = 4 * 24 * (current_seq_len + 512 - current_shared_len) * current_shared_len * 128
        flop_total = flop_local_attn + flop_shared_attn
        gflop = flop_total / 1e9  
        print(f"seq_len: {current_seq_len}, shared_len: {current_shared_len}, num_tile: {current_num_tile}, gflop: {gflop:.1f}")
        
