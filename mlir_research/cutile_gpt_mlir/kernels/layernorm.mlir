// SPDX-License-Identifier: Apache-2.0
//
// LayerNorm kernel in CUDA Tile IR
//
// This is a TRUE Tile-style implementation:
// - Declarative: describes WHAT to compute
// - Compiler-driven: HOW is determined by compiler
// - No manual block indexing or offset calculations
//
// Compare with Python version (cutile_gpt/kernels/layernorm.py)
// to see the difference between "Tile API usage" and "Tile philosophy"

cuda_tile.module @layernorm_module {
    // LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
    //
    // Input:
    //   x_ptr: Input tensor pointer (batch * seq * n_embd)
    //   gamma_ptr: Scale parameter (n_embd,)
    //   beta_ptr: Bias parameter (n_embd,)
    //   y_ptr: Output tensor pointer
    //   n_embd: Embedding dimension (tile size)
    //   eps: Small constant for numerical stability
    //
    // Note: This kernel processes ONE sequence element at a time
    //       Grid should be launched with (batch * seq) blocks
    cuda_tile.entry @layernorm_kernel(
        %x_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %gamma_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %beta_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %y_ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %n_embd_scalar: !cuda_tile.tile<i32>,
        %eps_scalar: !cuda_tile.tile<f32>
    ) {
        // === Step 1: Load input tile ===
        // Declarative: "Load n_embd elements from x_ptr"
        // Compiler decides: which threads, what memory pattern
        %init_token = make_token : !cuda_tile.token

        // Create offsets for loading
        %offsets = iota : !cuda_tile.tile<256xi32>

        // Broadcast pointer to tensor
        %x_ptr_reshaped = reshape %x_ptr
            : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<1x!cuda_tile.ptr<f32>>
        %x_ptr_broadcast = broadcast %x_ptr_reshaped
            : !cuda_tile.tile<1x!cuda_tile.ptr<f32>> -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        // Offset pointers
        %x_ptrs = offset %x_ptr_broadcast, %offsets
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>, !cuda_tile.tile<256xi32>
            -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        // Load data
        %x, %token1 = load_ptr_tko weak %x_ptrs token=%init_token
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<256xf32>, !cuda_tile.token

        // === Step 2: Compute mean ===
        // Declarative: "Reduce sum, then divide"
        // No manual thread coordination!
        %sum = reduce %x dim=0 identities=[0.0 : f32]
            : !cuda_tile.tile<256xf32> -> !cuda_tile.tile<f32>
        (%x_elem: !cuda_tile.tile<f32>, %acc: !cuda_tile.tile<f32>) {
            %new_acc = addf %x_elem, %acc : !cuda_tile.tile<f32>
            yield %new_acc : !cuda_tile.tile<f32>
        }

        // Convert n_embd to float for division
        %n_embd_f32 = itof %n_embd_scalar signed : !cuda_tile.tile<i32> -> !cuda_tile.tile<f32>

        // mean = sum / n_embd
        %mean = divf %sum, %n_embd_f32 rounding<approx>
            : !cuda_tile.tile<f32>

        // Broadcast mean for element-wise ops
        %mean_broadcast = broadcast %mean
            : !cuda_tile.tile<f32> -> !cuda_tile.tile<256xf32>

        // === Step 3: Compute variance ===
        // x_centered = x - mean
        %x_centered = subf %x, %mean_broadcast : !cuda_tile.tile<256xf32>

        // x_centered_sq = x_centered^2
        %x_centered_sq = mulf %x_centered, %x_centered
            : !cuda_tile.tile<256xf32>

        // var = sum(x_centered^2) / n_embd
        %var_sum = reduce %x_centered_sq dim=0 identities=[0.0 : f32]
            : !cuda_tile.tile<256xf32> -> !cuda_tile.tile<f32>
        (%sq_elem: !cuda_tile.tile<f32>, %acc: !cuda_tile.tile<f32>) {
            %new_acc = addf %sq_elem, %acc : !cuda_tile.tile<f32>
            yield %new_acc : !cuda_tile.tile<f32>
        }

        %var = divf %var_sum, %n_embd_f32 rounding<approx>
            : !cuda_tile.tile<f32>

        // === Step 4: Normalize ===
        // std = sqrt(var + eps)
        %eps_broadcast = broadcast %eps_scalar
            : !cuda_tile.tile<f32> -> !cuda_tile.tile<f32>
        %var_eps = addf %var, %eps_broadcast : !cuda_tile.tile<f32>
        %std = sqrt %var_eps : !cuda_tile.tile<f32>

        %std_broadcast = broadcast %std
            : !cuda_tile.tile<f32> -> !cuda_tile.tile<256xf32>

        // x_norm = (x - mean) / std
        %x_norm = divf %x_centered, %std_broadcast rounding<approx>
            : !cuda_tile.tile<256xf32>

        // === Step 5: Affine transform ===
        // Broadcast gamma and beta pointers
        %gamma_ptr_reshaped = reshape %gamma_ptr
            : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<1x!cuda_tile.ptr<f32>>
        %gamma_ptr_broadcast = broadcast %gamma_ptr_reshaped
            : !cuda_tile.tile<1x!cuda_tile.ptr<f32>> -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        %beta_ptr_reshaped = reshape %beta_ptr
            : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<1x!cuda_tile.ptr<f32>>
        %beta_ptr_broadcast = broadcast %beta_ptr_reshaped
            : !cuda_tile.tile<1x!cuda_tile.ptr<f32>> -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        // Load gamma and beta
        %gamma_ptrs = offset %gamma_ptr_broadcast, %offsets
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>, !cuda_tile.tile<256xi32>
            -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        %beta_ptrs = offset %beta_ptr_broadcast, %offsets
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>, !cuda_tile.tile<256xi32>
            -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        %gamma, %token2 = load_ptr_tko weak %gamma_ptrs token=%token1
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<256xf32>, !cuda_tile.token

        %beta, %token3 = load_ptr_tko weak %beta_ptrs token=%token2
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<256xf32>, !cuda_tile.token

        // y = x_norm * gamma + beta
        %y_scaled = mulf %x_norm, %gamma : !cuda_tile.tile<256xf32>
        %y = addf %y_scaled, %beta : !cuda_tile.tile<256xf32>

        // === Step 6: Store result ===
        %y_ptr_reshaped = reshape %y_ptr
            : !cuda_tile.tile<!cuda_tile.ptr<f32>> -> !cuda_tile.tile<1x!cuda_tile.ptr<f32>>
        %y_ptr_broadcast = broadcast %y_ptr_reshaped
            : !cuda_tile.tile<1x!cuda_tile.ptr<f32>> -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        %y_ptrs = offset %y_ptr_broadcast, %offsets
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>, !cuda_tile.tile<256xi32>
            -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        %token_final = store_ptr_tko weak %y_ptrs, %y token=%token3
            : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>, !cuda_tile.tile<256xf32>
            -> !cuda_tile.token

        return
    }

    // Helper: Broadcast pointer for gamma/beta
    // (Separated for clarity, could be inlined)
    cuda_tile.func @broadcast_ptr(
        %ptr: !cuda_tile.tile<!cuda_tile.ptr<f32>>
    ) -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>> {
        %ptr_reshaped = reshape %ptr
            : !cuda_tile.tile<!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<1x!cuda_tile.ptr<f32>>

        %ptr_broadcast = broadcast %ptr_reshaped
            : !cuda_tile.tile<1x!cuda_tile.ptr<f32>>
            -> !cuda_tile.tile<256x!cuda_tile.ptr<f32>>

        return %ptr_broadcast : !cuda_tile.tile<256x!cuda_tile.ptr<f32>>
    }
}

// Implementation Notes:
//
// 1. **Truly Declarative**
//    - No ct.bid() or manual block indexing
//    - No swizzle_2d() calculations
//    - Compiler handles thread-to-tile mapping
//
// 2. **Explicit Data Flow**
//    - Token-based ordering ensures correctness
//    - Load/store dependencies are clear
//
// 3. **Tile-level Operations**
//    - reduce: Automatically parallelized by compiler
//    - broadcast: Compiler decides optimal implementation
//    - No manual loop over elements
//
// 4. **Compare with Python version:**
//    Python (cutile_gpt/kernels/layernorm.py):
//      - ct.bid(), manual offsets
//      - Explicit loops and indexing
//      - Manual block/thread management
//
//    MLIR (this file):
//      - Pure tile operations
//      - Compiler-driven parallelization
//      - Hardware abstraction
//
// 5. **Limitations**
//    - Fixed tile size (256 here)
//    - Must match n_embd
//    - For variable sizes, need dynamic tile support
//
// This is what "Tile Philosophy" truly means!
