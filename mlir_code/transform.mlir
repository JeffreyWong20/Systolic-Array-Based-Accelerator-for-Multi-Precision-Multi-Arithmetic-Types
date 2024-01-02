// Batch size       = 128
// Input Dim        = 32
// Output Dim       = 32
func @linear_layer(%input_data: memref<128x32xf64>, %weight_matrix: memref<32x32xf64>, %bias: memref<32xf64>, %output: memref<32xf64>) {
    // Get the dimensions of the input data, weight matrix, and bias.
    %batch_size = affine.load %input_data[0] : memref<128x32xf64>                      
    %input_size = affine.load %weight_matrix[1] : memref<32x32xf64>
    %output_size = affine.load %bias[0] : memref<32xf64>

    // Perform matrix multiplication and bias addition using nested loops.
    affine.for %batch = 0 to %batch_size step block_size {
        affine.for %out_dim = 0 to %output_size step block_size{
            affine.for %in_dim = 0 to %input_size step channelA_height {
                channelA_1.load %input_data[%batch:batch*block_size, %in_dim:in_dim*block_size]
                channelA_2.load %weight_matrix[%out_dim:%out_dim*block_size, %in_dim:%in_dim*block_size]
                matmul
                accumulator.load %bias[%out_dim] : memref<32xf64>
                matsum
            }
            affine.store accumulator, %output[%batch:%batch+block_size, %out_dim:%out_dim+block_size] : memref<32xf64>
        }
    }
    return
}



func @linear_layer(%input_data: memref<128x32xf64>, %weight_matrix: memref<32x32xf64>, %bias: memref<32xf64>, %output: memref<32xf64>) {
    // Get the dimensions of the input data, weight matrix, and bias.
    %batch_size = affine.load %input_data[0] : memref<128x32xf64>                      
    %input_size = affine.load %weight_matrix[1] : memref<32x32xf64>
    %output_size = affine.load %bias[0] : memref<32xf64>

    // Perform matrix multiplication and bias addition using nested loops.
    affine.for %batch = 0 to %batch_size step block_size {
        affine.for %out_dim = 0 to %output_size step block_size{
            affine.for %in_dim = 0 to %input_size step channelA_height {
                channelA_1.load %input_data[%batch:batch*block_size, %in_dim:in_dim*channelA_height]
                channelA_2.load %weight_matrix[%out_dim:%out_dim*block_size, %in_dim:%in_dim*channelA_height]
                matmul
                accumulator.load %bias[] : memref<32xf64>
                matsum

            }
            affine.store accumulator, %output[%batch:%batch+block_size, %out_dim:%out_dim+block_size] : memref<32xf64>
        }
    }
    return
}


counter_1.load = 0
counter_2.load = 0
counter_3.load = 0


L1:
    counter_1.load 


L2:
    counter_2 = 


L3:
    counter_3 = 

