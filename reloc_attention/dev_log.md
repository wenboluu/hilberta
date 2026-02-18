# REMARK
    # 2025.7.25
        IMA 
            IMA problem has been solved, however the effectiveness and efficiency remain unresolved. 
                [Finding] For Efficiency, we have tried the cuda.stream(), however, it does not work by creating NaN which might be caused by the data conflict. Now we might turning to implement parallel computation in one kernel
                [ToDo] For Effectiveness, it probably because the reorder is not fully applied. However, it looks like a norm problem instead of a order problem. 
            ### [ToDo] Several Notes Regarding the kernel fusion
                1. The indexing for the offset and the base requires a revisit
                2. To make the code more elegant, probably we can have different instances to initialize different data pointer
                3. I still think it is a little bit wierd to add the shared_part offset on the indices dim to the batch and head dim

        Inpainting
            The mask and the image latent are concated and then feed to the transformer
                [ToDo] Technically we can directly apply our transformer, however, the performance might not be perfect, we can definitely try to have the masked latent to be the shared region together with the prompt token, however, this would degrade the latency
                [ToDo] Find a dataset
        
    # 2025.7.23
        [Wrong Finding] The relationship and the logic of the base, offset, and stride worth later revisiting. Previously in the normal K and V, the offsets=(0, GROUP_START * HEAD_DIM) which I think is not correct as it should not skip and feature dimension but should skip the seq_len dim which has already been done in the base(However, this might not be a consistent coding way). But after solving this part, the problem of IMA still remain. 