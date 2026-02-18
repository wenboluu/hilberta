# REMARK
    # 2025.7.25
        The IMA problem has been solved, however the effectiveness and efficiency remain unresolved. 
            For Efficiency, we have tried the cuda.stream(), however, it does not work by creating NaN which might be caused by the data conflict. Now we might turning to implement parallel computation in one kernel
            For Effectiveness, it probably because the reorder is not fully applied. However, it looks like a norm problem instead of a order problem. 
    # 2025.7.23
        The relationship and the logic of the base, offset, and stride worth later revisiting. Previously in the normal K and V, the offsets=(0, GROUP_START * HEAD_DIM) which I think is not correct as it should not skip and feature dimension but should skip the seq_len dim which has already been done in the base(However, this might not be a consistent coding way). But after solving this part, the problem of IMA still remain. 