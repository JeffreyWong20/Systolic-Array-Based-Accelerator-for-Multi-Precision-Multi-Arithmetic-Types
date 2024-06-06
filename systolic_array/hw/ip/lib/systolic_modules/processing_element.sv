//////////////////////////////////////////////////////////////////////////////////
// Engineer: 
// 
// Design Name: 
// Create Date:
// Module Name: processing_element
// Tool Versions: 
// Description: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module processing_element #(
    parameter PRECISION = top_pkg::FLOAT_32,
    parameter DATA_WIDTH = 32,
    parameter ACCUMULATOR_WIDTH = 32,
    parameter FRACTIONAL_BITS = 0,
    parameter FLOAT_WIDTH = 32
) (
    input  logic                            core_clk,
    input  logic                            resetn,

    input  logic                            pulse_systolic_module,

    input  logic                            pe_forward_in_valid,
    input  logic signed [DATA_WIDTH-1:0]    pe_forward_in,

    input  logic                            pe_down_in_valid,
    input  logic signed [DATA_WIDTH-1:0]    pe_down_in,
    
    output logic                            pe_forward_out_valid,
    output logic [DATA_WIDTH-1:0]           pe_forward_out,
    
    output logic                            pe_down_out_valid,
    output logic [DATA_WIDTH-1:0]           pe_down_out,

    input  logic                            bias_valid,
    input  logic signed [DATA_WIDTH-1:0]    bias,

    input  logic                                             activation_valid,
    input  logic [$bits(top_pkg::ACTIVATION_FUNCTION_e)-1:0] activation,

    input  logic                            shift_valid,
    input  logic [ACCUMULATOR_WIDTH-1:0]    shift_data,
    
    output logic [ACCUMULATOR_WIDTH-1:0]    pe_acc,

    input  logic [DATA_WIDTH-1:0]           layer_config_leaky_relu_alpha_value,

    output logic [DATA_WIDTH-1:0]           debug_update_counter
);

// ==================================================================================================================================================
// Declarations
// ==================================================================================================================================================

logic                               update_accumulator;

logic                               overwrite_accumulator;
logic [ACCUMULATOR_WIDTH-1:0]       overwrite_data;

logic                               bias_out_valid_comb;
logic                               bias_out_valid;
logic [ACCUMULATOR_WIDTH-1:0]       extended_bias;
logic [ACCUMULATOR_WIDTH-1:0]       pe_acc_add_bias; // TODO: The multi-precision system is messed up here

logic                               activated_feature_valid;
logic [ACCUMULATOR_WIDTH-1:0]       activated_feature; // TODO: The multi-precision system is messed up here
logic [ACCUMULATOR_WIDTH-1:0]       extended_layer_config_leaky_relu_alpha_value;

// ==================================================================================================================================================
// Accumulator
// ==================================================================================================================================================

mac #(
    .PRECISION          (PRECISION),
    .DATA_WIDTH         (DATA_WIDTH),
    .ACCUMULATOR_WIDTH  (ACCUMULATOR_WIDTH)
) mac_i (
    .core_clk,            
    .resetn,
    
    .in_valid           (update_accumulator),
    .in_ready           (),

    .a                  (pe_forward_in),
    .b                  (pe_down_in),

    .overwrite          (overwrite_accumulator),
    .overwrite_data     (overwrite_data),
    
    .accumulator        (pe_acc)
);

// Bias addition
// -------------------------------------------------------------
logic floating_point_valid; // TODO: Remove this later

assign extended_bias = {{{ACCUMULATOR_WIDTH-FRACTIONAL_BITS} {bias[DATA_WIDTH-1]}}, bias,{FRACTIONAL_BITS{1'd0}}};
assign extended_layer_config_leaky_relu_alpha_value = {{{ACCUMULATOR_WIDTH-FRACTIONAL_BITS} {layer_config_leaky_relu_alpha_value[DATA_WIDTH-1]}}, layer_config_leaky_relu_alpha_value,{FRACTIONAL_BITS{1'd0}}};

if (PRECISION == top_pkg::FLOAT_32) begin
    
    assign floating_point_valid = 1;
    // fp_add bias_adder (
    // .s_axis_a_tvalid              (1'b1),
    // .s_axis_a_tdata               (pe_acc),

    // .s_axis_b_tvalid              (bias_valid),
    // .s_axis_b_tdata               (bias),

    // .m_axis_result_tvalid         (bias_out_valid_comb),
    // .m_axis_result_tdata          (pe_acc_add_bias_comb)
    // );

    // always_ff @(posedge core_clk or negedge resetn) begin
    //     if (!resetn) begin
    //         bias_out_valid <= '0;
    //         pe_acc_add_bias <= '0;

    //     end else begin
    //         bias_out_valid <= bias_out_valid_comb;
    //         pe_acc_add_bias <= pe_acc_add_bias_comb;
    //     end
    // end

end else begin
    // Fixed point
    always_comb begin
        bias_out_valid = bias_valid;
        pe_acc_add_bias = pe_acc + extended_bias;
    end
end

// Activations
// -------------------------------------------------------------

activation_core #(
    .PRECISION  (PRECISION),
    .DATA_WIDTH (ACCUMULATOR_WIDTH),
    .FLOAT_WIDTH(FLOAT_WIDTH)
) activation_core_i (
    .core_clk                            (core_clk),
    .resetn                              (resetn),
    .sel_activation                      (activation),

    .in_feature_valid (activation_valid),
    .in_feature       (pe_acc),

    .activated_feature_valid (activated_feature_valid),
    .activated_feature       (activated_feature),

    .layer_config_leaky_relu_alpha_value (extended_layer_config_leaky_relu_alpha_value)
);

// ==================================================================================================================================================
// Logic
// ==================================================================================================================================================

assign update_accumulator = pulse_systolic_module && pe_forward_in_valid && pe_down_in_valid;

// Register incoming (forward/down) features
always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        pe_forward_out_valid        <= '0;
        pe_forward_out              <= '0;
        
        pe_down_out_valid           <= '0;
        pe_down_out                 <= '0;

    end else if (pulse_systolic_module) begin
        pe_forward_out_valid        <= pe_forward_in_valid;
        pe_forward_out              <= pe_forward_in;

        pe_down_out_valid           <= pe_down_in_valid;
        pe_down_out                 <= pe_down_in;

    end
end

// Overwrite accumulator for activation, bias and shifting
always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        overwrite_accumulator <= '0;
        overwrite_data <= '0;

    end else begin
        overwrite_accumulator <= bias_out_valid || activated_feature_valid || shift_valid;

        overwrite_data        <= bias_out_valid ? pe_acc_add_bias
                                : activated_feature_valid ? activated_feature
                                : shift_valid ? shift_data
                                : overwrite_data;
    end
end

`ifdef DEBUG

always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        debug_update_counter <= '0;

    end else if (update_accumulator) begin
        debug_update_counter <= debug_update_counter + 1'b1;

    end
end

`endif

// ======================================================================================================
// Assertions
// ======================================================================================================

// TO DO: fix
// P_update_acc_both_valid: assert property (
//     @(posedge core_clk) disable iff (!resetn)
//     (!pe_forward_in_valid || !pe_down_in_valid) |=> pe_acc == $past(pe_acc, 1)
// );

endmodule