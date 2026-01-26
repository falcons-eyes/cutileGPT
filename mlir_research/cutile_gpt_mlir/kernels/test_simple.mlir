// Simple test kernel for CUDA Tile
cuda_tile.module @test_simple {
    cuda_tile.entry @add_kernel(
        %a_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %b_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %c_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>
    ) {
        // Create a token for memory operations
        %token_init = cuda_tile.make_token : !cuda_tile.token

        // Load values
        %a, %token1 = cuda_tile.load_ptr_tko weak %a_ptr token=%token_init
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<f32>, !cuda_tile.token

        %b, %token2 = cuda_tile.load_ptr_tko weak %b_ptr token=%token1
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<f32>, !cuda_tile.token

        // Add
        %c = cuda_tile.addf %a, %b : !cuda_tile.tile<f32>

        // Store result
        %token3 = cuda_tile.store_ptr_tko weak %c_ptr, %c token=%token2
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>, !cuda_tile.tile<f32>
            -> !cuda_tile.token

        cuda_tile.return
    }
}
