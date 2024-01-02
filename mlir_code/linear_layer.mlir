// Batch size       = 128
// Input Dim        = 32
// Output Dim       = 32
func @linear_layer(%input_data: memref<128x32xf64>, %weight_matrix: memref<32x32xf64>, %bias: memref<32xf64>, %output: memref<32xf64>) {
  // Get the dimensions of the input data, weight matrix, and bias.
  %batch_size = affine.load %input_data[0] : memref<128x32xf64>                      
  %input_size = affine.load %weight_matrix[1] : memref<32x32xf64>
  %output_size = affine.load %bias[0] : memref<32xf64>

  // Perform matrix multiplication and bias addition using nested loops.
  affine.for %batch = 0 to %batch_size {
    affine.for %out_dim = 0 to %output_size {
      %acc = constant 0.0 : f64
      affine.for %in_dim = 0 to %input_size {
        %input_val = affine.load %input_data[%batch, %in_dim] : memref<128x32xf64>
        %weight_val = affine.load %weight_matrix[%out_dim, %in_dim] : memref<32x32xf64>
        %mul = mulf %input_val, %weight_val : f64
        %acc = addf %acc, %mul : f64
      }
      %bias_val = affine.load %bias[%out_dim] : memref<32xf64>
      %output_val = addf %acc, %bias_val : f64
      affine.store %output_val, %output[%batch, %out_dim] : memref<32xf64>
    }
  }
  return
}


// Translation into systolic array behaviour



1. Given that the dimension of the systolic array is fixed by N*N
2. Buffer a, b has dimension N * K

The systolic array can perform the matrix multiplication on N cycle
So any loop that can be mapped as a matrix multiplication can be specified in one instruction.
With the largest dimension of a & b to be bounded by N

// What do we want to do?
// We have a block that is good at performing matrix multiplication.
// We have auto optim


It is the job for compiler to partition the m1 and m2 to different sizes of matrix multiplication and call this custom command.
Matmul destination a b

Load %A[:, :] a
Load %B[:, :] b
Matmul c a b
Store c %c[:,:]




// Compiler should be able to manage this 
To Perform an arbitary matmul m1 = X*Y m2 = Y*Z
For Y > N, we would need 
For Y < N, we would need 




// C += A * B.
func @matmul(%A: memref<2048x2048xf64>, %B: memref<2048x2048xf64>, %C: memref<2048x2048xf64>) {
  affine.for %arg3 = 0 to 2048 {
    affine.for %arg4 = 0 to 2048 {
     affine.for %arg5 = 0 to 2048 {
       %a = affine.load %A[%arg3, %arg5] : memref<2048x2048xf64>
       %b = affine.load %B[%arg5, %arg4] : memref<2048x2048xf64>
       %ci = affine.load %C[%arg3, %arg4] : memref<2048x2048xf64>
       %p = mulf %a, %b : f64
       %co = addf %ci, %p : f64
       affine.store %co, %C[%arg3, %arg4] : memref<2048x2048xf64>
     }
} }
return
}


func @main() {
  %A = alloc() : memref<2048x2048xf64>
  %B = alloc() : memref<2048x2048xf64>
  %C = alloc() : memref<2048x2048xf64>
  %cf1 = constant 1.00000e+00 : f64
  linalg.fill(%A, %cf1) : memref<2048x2048xf64>, f64
  linalg.fill(%B, %cf1) : memref<2048x2048xf64>, f64
  linalg.fill(%C, %cf1) : memref<2048x2048xf64>, f64
  call @matmul(%A, %B, %C) : (memref<2048x2048xf64>, memref<2048x2048xf64>, memref<2048x2048xf64>) -> ()
  call @print_memref_2d_f64(%C): (memref<2048x2048xf64>) -> ()
  return
}
func @print_memref_2d_f64(memref<2048x2048xf64>)