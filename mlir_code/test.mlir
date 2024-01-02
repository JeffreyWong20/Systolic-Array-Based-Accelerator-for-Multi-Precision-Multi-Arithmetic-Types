// C += A * B + bias.
func @matmul_with_bias(%A: memref<2048x2048xf64>, %B: memref<2048x2048xf64>, %C: memref<2048x2048xf64>, %bias: memref<2048x2048xf64>) {
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
      affine.for %arg5 = 0 to 2048 {
        %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf64>
        %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf64>
        %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf64>
        %p = mulf %a, %b : f64
        %co = addf %ci, %p : f64
        %bias_val = affine.load %bias[%arg3, %arg4] : memref<2048x2048xf64>
        %co_with_bias = addf %co, %bias_val : f64
        affine.store %co_with_bias, %C[%arg3, %arg4] : memref<2048x2048xf64>
      }
    }
  }
  return
}
