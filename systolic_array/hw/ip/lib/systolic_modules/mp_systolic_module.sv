//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:
// Design Name: 
// Module Name: systolic_module
// Project Name:
// Target Devices:
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module mp_systolic_module #(
    parameter HIGH_PRECISION_DATA_WIDTH = top_pkg::HIGH_PRECISION_DATA_WIDTH,
    parameter LOW_PRECISION_DATA_WIDTH = top_pkg::LOW_PRECISION_DATA_WIDTH,
    parameter HIGH_PRECISION_FRACTIONAL_BITS = top_pkg::HIGH_PRECISION_FRACTIONAL_BITS,
    parameter LOW_PRECISION_FRACTIONAL_BITS = top_pkg::LOW_PRECISION_FRACTIONAL_BITS,
    parameter HIGH_PRECISION_ACCUMULATOR_WIDTH = top_pkg::HIGH_PRECISION_ACCUMULATOR_WIDTH,
    parameter LOW_PRECISION_ACCUMULATOR_WIDTH = top_pkg::LOW_PRECISION_ACCUMULATOR_WIDTH,
    parameter HIGH_PRECISION_PE_COUNT = top_pkg::HIGH_PRECISION_PE_COUNT,
    parameter LOW_PRECISION_PE_COUNT = top_pkg::LOW_PRECISION_PE_COUNT,


    parameter PRECISION = top_pkg::FLOAT_32,
    parameter FLOAT_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter PASS_THROUGH_DATA_WIDTH = 32,
    parameter ACCUMULATOR_WIDTH = 32,
    parameter FRACTIONAL_BITS = 0,
    parameter MATRIX_N = 4
) (
    input  logic                                                 core_clk,            
    input  logic                                                 resetn,

    input  logic                                                 pulse_systolic_module,
    
    input  logic [MATRIX_N-1:0]                                  sys_module_forward_in_valid,
    input  logic [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_forward_in,
    input  logic [MATRIX_N-1:0] [PASS_THROUGH_DATA_WIDTH-1:0]    sys_module_forward_in_pass,
    
    input  logic [MATRIX_N-1:0]                                  sys_module_down_in_valid,
    input  logic [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_down_in,

    output logic [MATRIX_N-1:0]                                  sys_module_forward_out_valid,
    output logic [MATRIX_N-1:0] [LOW_PRECISION_DATA_WIDTH-1:0]   sys_module_forward_out,
    output logic [MATRIX_N-1:0] [PASS_THROUGH_DATA_WIDTH-1:0]    sys_module_forward_out_pass,
    
    output logic [MATRIX_N-1:0]                                  sys_module_down_out_valid,
    output logic [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_down_out,

    input  logic                                                 bias_valid,
    input  logic [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  bias,

    input  logic                                                 activation_valid,
    input  logic [$bits(top_pkg::ACTIVATION_FUNCTION_e)-1:0]     activation,
    
    input  logic                                                 shift_valid,

    // Accumulators for each Processing Element, from which output matrix can be constructed
    // One more row than required to shift in zeros into last row during SHIFT phase
    output logic [MATRIX_N:0] [MATRIX_N-1:0] [ACCUMULATOR_WIDTH-1:0]    sys_module_pe_acc,

    output logic                                                        diagonal_flush_done,

    input logic [DATA_WIDTH-1:0]                                        layer_config_leaky_relu_alpha_value,

    output logic [MATRIX_N-1:0] [MATRIX_N-1:0] [DATA_WIDTH-1:0]         debug_update_counter
);


// ============================================================================================
// Declarations
// ============================================================================================

//   <    row    > <    col   > <      data      >
logic [MATRIX_N-1:0] [HIGH_PRECISION_PE_COUNT:0] [0:0]                             sys_module_pe_forward_high_valid;
logic [MATRIX_N-1:0] [HIGH_PRECISION_PE_COUNT:0] [HIGH_PRECISION_DATA_WIDTH-1:0]   sys_module_pe_forward_high;
logic [MATRIX_N-1:0] [MATRIX_N:0] [PASS_THROUGH_DATA_WIDTH-1:0]                    sys_module_pe_forward_copy;

logic [MATRIX_N-1:0] [LOW_PRECISION_PE_COUNT:0] [0:0]                             sys_module_pe_forward_low_valid;
logic [MATRIX_N-1:0] [LOW_PRECISION_PE_COUNT:0] [LOW_PRECISION_DATA_WIDTH-1:0]    sys_module_pe_forward_low;

//   <    row    > <    col   > <      data      >
logic [MATRIX_N:0] [HIGH_PRECISION_PE_COUNT-1:0] [0:0]                              sys_module_pe_down_high_valid;
logic [MATRIX_N:0] [HIGH_PRECISION_PE_COUNT-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]    sys_module_pe_down_high;
logic [MATRIX_N:0] [LOW_PRECISION_PE_COUNT-1:0] [0:0]                               sys_module_pe_down_low_valid;
logic [MATRIX_N:0] [LOW_PRECISION_PE_COUNT-1:0] [LOW_PRECISION_DATA_WIDTH-1:0]      sys_module_pe_down_low;

logic [HIGH_PRECISION_PE_COUNT-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]                bias_high;
logic [LOW_PRECISION_PE_COUNT-1:0] [LOW_PRECISION_DATA_WIDTH-1:0]                  bias_low;


logic [MATRIX_N:0] [HIGH_PRECISION_PE_COUNT-1:0] [HIGH_PRECISION_ACCUMULATOR_WIDTH-1:0]    sys_module_pe_acc_high;
logic [MATRIX_N:0] [LOW_PRECISION_PE_COUNT-1:0] [LOW_PRECISION_ACCUMULATOR_WIDTH-1:0]     sys_module_pe_acc_low;

logic [MATRIX_N-1:0] forward_flush_done;
logic [MATRIX_N-1:0] down_flush_done;



// ============================================================================================
// Instances
// ============================================================================================


for (genvar col = 0; col < HIGH_PRECISION_PE_COUNT; col++) begin : high_cols_gen
    fixed_cast_single #(
        .IN_WIDTH (HIGH_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*HIGH_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_high_row_0_i (
        .data_in    (sys_module_pe_acc_high[0][col]),
        .data_out   (sys_module_pe_acc[0][col])
    );
    always_comb begin 
        sys_module_pe_down_high_valid   [0][col] = sys_module_down_in_valid[col];
        sys_module_pe_down_high         [0][col] = sys_module_down_in[col][HIGH_PRECISION_DATA_WIDTH:0];

        sys_module_down_out_valid [col] = sys_module_pe_down_high_valid [MATRIX_N] [col];
        sys_module_down_out       [col] = sys_module_pe_down_high [MATRIX_N] [col];
    end 
    always_comb begin 
        bias_high [col] = bias[col][HIGH_PRECISION_DATA_WIDTH:0];
    end
end : high_cols_gen


for (genvar col = 0; col < LOW_PRECISION_PE_COUNT; col++) begin : low_cols_gen
    fixed_cast_single #(
        .IN_WIDTH (LOW_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*LOW_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_low_row_0_i (
        .data_in    (sys_module_pe_acc_low[0][col]),
        .data_out   (sys_module_pe_acc[0][HIGH_PRECISION_PE_COUNT+col])
    );
    always_comb begin 
        sys_module_pe_down_low_valid    [0][col] = sys_module_down_in_valid[HIGH_PRECISION_PE_COUNT+col];
        sys_module_pe_down_low          [0][col] = sys_module_down_in[HIGH_PRECISION_PE_COUNT+col][LOW_PRECISION_DATA_WIDTH:0];

        sys_module_down_out_valid [HIGH_PRECISION_PE_COUNT+col] = sys_module_pe_down_low_valid [MATRIX_N] [col];
        sys_module_down_out       [HIGH_PRECISION_PE_COUNT+col] = sys_module_pe_down_low [MATRIX_N] [col];
    end
    always_comb begin 
        bias_low [col] = bias[HIGH_PRECISION_PE_COUNT+col][LOW_PRECISION_DATA_WIDTH-1:0];
    end
    
end : low_cols_gen


// for (genvar col=0; col < MATRIX_N; col++) begin
    
//     always_comb begin
//         // Drive down inputs
//         sys_module_pe_down            [0][col] = sys_module_down_in      [col];
//         sys_module_pe_down_valid      [0][col] = sys_module_down_in_valid[col];

//         // Drive down outputs
//         sys_module_down_out_valid [col] = sys_module_pe_down_valid [MATRIX_N] [col];
//         sys_module_down_out       [col] = sys_module_pe_down [MATRIX_N] [col];
//     end
// end


for (genvar row = 0; row < MATRIX_N; row++) begin : rows_gen
    for (genvar col = 0; col < HIGH_PRECISION_PE_COUNT; col++) begin : high_pe_cols_gen
        processing_element #(
            .PRECISION          (PRECISION),
            .DATA_WIDTH         (HIGH_PRECISION_DATA_WIDTH),
            .ACCUMULATOR_WIDTH  (HIGH_PRECISION_ACCUMULATOR_WIDTH),
            .FRACTIONAL_BITS    (HIGH_PRECISION_FRACTIONAL_BITS),
            .PASS_THROUGH_DATA_WIDTH (PASS_THROUGH_DATA_WIDTH),
            .FLOAT_WIDTH        (FLOAT_WIDTH)
        ) pe_i (
            .core_clk,
            .resetn,

            .pulse_systolic_module      (pulse_systolic_module),

            .pe_forward_in_valid        (sys_module_pe_forward_high_valid      [row]   [col]   ),
            .pe_forward_in              (sys_module_pe_forward_high            [row]   [col]   ),
            .pe_forward_in_copy         (sys_module_pe_forward_copy            [row]   [col]   ),
            
            .pe_down_in_valid           (sys_module_pe_down_high_valid         [row]   [col]   ),
            .pe_down_in                 (sys_module_pe_down_high               [row]   [col]   ),
            
            .pe_forward_out_valid       (sys_module_pe_forward_high_valid      [row]   [col+1] ),
            .pe_forward_out             (sys_module_pe_forward_high            [row]   [col+1] ),
            .pe_forward_out_copy        (sys_module_pe_forward_copy            [row]   [col+1] ),
            
            .pe_down_out_valid          (sys_module_pe_down_high_valid         [row+1] [col] ),
            .pe_down_out                (sys_module_pe_down_high               [row+1] [col] ),

            .bias_valid                 (bias_valid),
            .bias                       (bias_high[col]),

            .activation_valid           (activation_valid),
            .activation                 (activation),

            .shift_valid                (shift_valid                                     ),
            .shift_data                 (sys_module_pe_acc_high           [row+1]   [col]),

            .pe_acc                     (sys_module_pe_acc_high           [row]   [col]   ),

            .layer_config_leaky_relu_alpha_value (layer_config_leaky_relu_alpha_value),

            .debug_update_counter       (debug_update_counter[row][col])
        );

    end : high_pe_cols_gen

    for (genvar col = 0; col < LOW_PRECISION_PE_COUNT; col++) begin : low_pe_cols_gen
        processing_element #(
            .PRECISION          (PRECISION),
            .DATA_WIDTH         (LOW_PRECISION_DATA_WIDTH),
            .ACCUMULATOR_WIDTH  (LOW_PRECISION_ACCUMULATOR_WIDTH),
            .FRACTIONAL_BITS    (LOW_PRECISION_FRACTIONAL_BITS),
            .PASS_THROUGH_DATA_WIDTH (PASS_THROUGH_DATA_WIDTH),
            .FLOAT_WIDTH        (FLOAT_WIDTH)
        ) pe_i (
            .core_clk,
            .resetn,

            .pulse_systolic_module      (pulse_systolic_module),

            .pe_forward_in_valid        (sys_module_pe_forward_low_valid       [row]   [col]   ),
            .pe_forward_in              (sys_module_pe_forward_low             [row]   [col]   ),
            .pe_forward_in_copy         (sys_module_pe_forward_copy            [row]   [LOW_PRECISION_PE_COUNT+col]   ),
            
            .pe_down_in_valid           (sys_module_pe_down_low_valid          [row]   [col]  ),
            .pe_down_in                 (sys_module_pe_down_low                [row]   [col]   ),
            
            .pe_forward_out_valid       (sys_module_pe_forward_low_valid       [row]   [col+1] ),
            .pe_forward_out             (sys_module_pe_forward_low             [row]   [col+1] ),
            .pe_forward_out_copy        (sys_module_pe_forward_copy            [row]   [LOW_PRECISION_PE_COUNT+col+1] ),
            
            .pe_down_out_valid          (sys_module_pe_down_low_valid          [row+1] [col] ),
            .pe_down_out                (sys_module_pe_down_low                [row+1] [col] ),

            .bias_valid                 (bias_valid),
            .bias                       (bias_low[col]),

            .activation_valid           (activation_valid),
            .activation                 (activation),

            .shift_valid                (shift_valid                                        ),
            .shift_data                 (sys_module_pe_acc_low                [row+1]   [col]),

            .pe_acc                     (sys_module_pe_acc_low               [row]   [col]   ),

            .layer_config_leaky_relu_alpha_value (layer_config_leaky_relu_alpha_value),

            .debug_update_counter       (debug_update_counter[row][LOW_PRECISION_PE_COUNT+col])
        );
    end : low_pe_cols_gen
end : rows_gen


for (genvar casting_row = 0; casting_row < MATRIX_N; casting_row++) begin
    fixed_cast_single #(
        .IN_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .IN_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (LOW_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (LOW_PRECISION_FRACTIONAL_BITS)
    ) casting_row_i (
        .data_in    (sys_module_pe_forward_high[casting_row][HIGH_PRECISION_PE_COUNT]),
        .data_out   (sys_module_pe_forward_low[casting_row][0])
    );
    always_comb begin
        sys_module_pe_forward_low_valid[casting_row][0] = sys_module_pe_forward_high_valid[casting_row][HIGH_PRECISION_PE_COUNT];
        // sys_module_forward_high_pass[casting_row][0] = sys_module_pe_forward_high[casting_row][HIGH_PRECISION_PE_COUNT];
    end
end


// Input to lowest row during SHIFT phase
assign sys_module_pe_acc_low [MATRIX_N] = '0;
assign sys_module_pe_acc_high [MATRIX_N] = '0;

// ============================================================================================
// Logic
// ============================================================================================

for (genvar row=0; row < MATRIX_N; row++) begin
    always_comb begin
        // Drive forward inputs
        sys_module_pe_forward_high          [row][0] = sys_module_forward_in      [row];
        sys_module_pe_forward_high_valid    [row][0] = sys_module_forward_in_valid[row];
        sys_module_pe_forward_copy          [row][0] = sys_module_forward_in_pass [row];

        // Drive forward outputs
        sys_module_forward_out_valid [row]  = sys_module_pe_forward_low_valid [row] [LOW_PRECISION_PE_COUNT];
        sys_module_forward_out [row]        = sys_module_pe_forward_low [row] [LOW_PRECISION_PE_COUNT];
        sys_module_forward_out_pass [row]   = sys_module_pe_forward_copy [row] [MATRIX_N];
    end
end

assign forward_flush_done = ~sys_module_forward_out_valid;

assign down_flush_done = ~sys_module_down_out_valid;

assign diagonal_flush_done = &forward_flush_done && &down_flush_done;

endmodule