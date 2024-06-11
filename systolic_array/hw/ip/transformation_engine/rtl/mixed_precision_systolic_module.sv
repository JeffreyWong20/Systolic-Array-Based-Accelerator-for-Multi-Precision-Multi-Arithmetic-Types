
import top_pkg::*;

module mixed_precision_systolic_module #(
    parameter PRECISION = top_pkg::FLOAT_32,
    parameter FLOAT_WIDTH = 32,

    parameter MATRIX_N = top_pkg::TRANSFORMATION_ROWS,
    parameter SYSTOLIC_MODULE_COUNT = 32
) (
    input logic                                                                                       core_clk,
    input logic                                                                                       resetn,

    input  logic                                                                                      weight_channel_resp_valid,
    input  WEIGHT_CHANNEL_RESP_t                                                                      weight_channel_resp,

    input  [MATRIX_N-1:0]                                     mp_sys_module_forward_high_valid, // 16
    input  [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]     mp_sys_module_forward_high, // input is always in high precision
    output [MATRIX_N-1:0]                                     mp_sys_module_forward_out_high_valid,
    output [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]     mp_sys_module_forward_out_high,

    input  logic [SYSTOLIC_MODULE_COUNT*MATRIX_N-1:0] [31:0]                                          layer_config_bias_value,
    input  logic [1:0]                                                                                layer_config_activation_function_value,
    input  logic [31:0]                                                                               layer_config_leaky_relu_alpha_value,

    // output [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_pe_acc_high_casted, // NOTE: Data will get writeback in a high precision format. Low precision will get padded back to high precision by using 0.
    // output [LOW_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_pe_acc_low_casted, // NOTE: Data will get writeback in a high precision format. Low precision will get padded back to high precision by using 0.
    output [SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0] sys_module_pe_acc_casted,

    input                                                                           pulse_systolic_module,
    input                                                                           shift_sys_module,
    input                                                                           bias_valid,
    input                                                                           activation_valid,

    // Writeback AXI_DATA_WIDTH shifting logic (shifting horizontally)
    output [SYSTOLIC_MODULE_COUNT-1:0] sys_module_active,                                      // use to recall if a systolic array is active or not to write out the result
    output [$clog2(SYSTOLIC_MODULE_COUNT)-1:0] sys_module_active_count,
    input  clean_sys_module_active

);


 // Driven from weight channel
logic [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0]                                  sys_module_forward_high_valid; // 16
logic [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_forward_high; // input is always in high precision

logic [LOW_PRECISION_SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0]                                  sys_module_forward_low_valid; // 16
logic [LOW_PRECISION_SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0] [LOW_PRECISION_DATA_WIDTH-1:0]   sys_module_forward_low; // input is always in high precision
logic [LOW_PRECISION_SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_forward_high_pass; // only used in low precision systolic array

// Driven from weight channel
logic [MAX_FEATURE_COUNT-1:0]                                                   sys_module_down_in_high_valid;
logic [MAX_FEATURE_COUNT-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]                   sys_module_down_in_high;
logic [MAX_FEATURE_COUNT-1:0]                                                   sys_module_down_in_low_valid;
logic [MAX_FEATURE_COUNT-1:0] [LOW_PRECISION_DATA_WIDTH-1:0]                    sys_module_down_in_low;

logic [MAX_FEATURE_COUNT-1:0]                                                   sys_module_down_out_high_valid;
logic [MAX_FEATURE_COUNT-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]                   sys_module_down_out_high;

logic [MAX_FEATURE_COUNT-1:0]                                                   sys_module_down_out_low_valid;
logic [MAX_FEATURE_COUNT-1:0] [LOW_PRECISION_DATA_WIDTH-1:0]                    sys_module_down_out_low;

logic [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT*MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]                     sys_bias_per_row_high;
logic [LOW_PRECISION_SYSTOLIC_MODULE_COUNT*MATRIX_N-1:0] [LOW_PRECISION_DATA_WIDTH-1:0]                       sys_bias_per_row_low;

logic [SYSTOLIC_MODULE_COUNT-1:0]                                               sys_module_flush_done;
logic [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N:0] [MATRIX_N-1:0] [HIGH_PRECISION_ACCUMULATOR_WIDTH-1:0]  sys_module_pe_acc_high; // NOTE: Data will get writeback in a high precision format. Low precision will get padded back to high precision by using 0.
logic [LOW_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N:0] [MATRIX_N-1:0] [LOW_PRECISION_ACCUMULATOR_WIDTH-1:0]  sys_module_pe_acc_low; // NOTE: Data will get writeback in a high precision format. Low precision will get padded back to high precision by using 0.

logic [SYSTOLIC_MODULE_COUNT-1:0] sys_module_active_m;                                  // use to recall if a systolic array is active or not to write out the result

assign sys_module_active = sys_module_active_m;
// ==================================================================================================================================================
logic [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0] debug_update_counter_high;
logic [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0] debug_update_counter_inv_high;
logic [LOW_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [MATRIX_N-1:0] [LOW_PRECISION_DATA_WIDTH-1:0] debug_update_counter_low;

assign debug_update_counter_inv = ~debug_update_counter_high;

parameter HIGH_PRECISION_DATA_WIDTH = top_pkg::HIGH_PRECISION_DATA_WIDTH;
parameter LOW_PRECISION_DATA_WIDTH = top_pkg::LOW_PRECISION_DATA_WIDTH;
parameter HIGH_PRECISION_ACCUMULATOR_WIDTH = top_pkg::HIGH_PRECISION_ACCUMULATOR_WIDTH;
parameter LOW_PRECISION_ACCUMULATOR_WIDTH = top_pkg::LOW_PRECISION_ACCUMULATOR_WIDTH;

count_ones #(
    .INPUT_WIDTH(SYSTOLIC_MODULE_COUNT)
) number_of_active_systolic_module (
  .data(sys_module_active),
  .count(sys_module_active_count)
);


always_comb begin
    sys_module_forward_high_valid [0] = mp_sys_module_forward_high_valid;
end
for (genvar index = 0; index < MATRIX_N; index++) begin
    always_comb begin
        sys_module_forward_high   [0][index] = mp_sys_module_forward_high [index];
    end
end

// ==================================================================================================================================================
// Instances
// ==================================================================================================================================================
// High precision systolic array
for (genvar sys_module = 0; sys_module < HIGH_PRECISION_SYSTOLIC_MODULE_COUNT; sys_module++) begin
    // Driving from weight channel
    always_comb begin
        sys_module_down_in_high_valid [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] = {MATRIX_N{weight_channel_resp_valid}} & weight_channel_resp.valid_mask[sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N];
    end

    always_ff @(posedge core_clk or negedge resetn) begin
        if (!resetn | clean_sys_module_active) begin
            // sys_module_active_m [sys_module+1 : sys_module] <= '0;
            sys_module_active_m [sys_module] <= '0;

        end else begin
            // sys_module_active_m [sys_module+1 : sys_module] <= |sys_module_down_in_high_valid[sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] | sys_module_active_m [sys_module+1 : sys_module];
            sys_module_active_m [sys_module] <= |sys_module_down_in_high_valid[sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] | sys_module_active_m [sys_module];
        end
    end

    for (genvar index = sys_module*MATRIX_N; index < (sys_module*MATRIX_N + (MATRIX_N-1) + 1); index++) begin
        always_comb begin
            sys_module_down_in_high       [index] = weight_channel_resp.data[index][HIGH_PRECISION_DATA_WIDTH-1:0];
            sys_bias_per_row_high         [index] = layer_config_bias_value[index][HIGH_PRECISION_DATA_WIDTH-1:0];
        end
    end

    systolic_module #(
        .PRECISION (PRECISION),
        .FLOAT_WIDTH (FLOAT_WIDTH),
        .ACCUMULATOR_WIDTH(HIGH_PRECISION_ACCUMULATOR_WIDTH),
        .DATA_WIDTH  (HIGH_PRECISION_DATA_WIDTH),
        .FRACTIONAL_BITS (HIGH_PRECISION_FRACTIONAL_BITS),
        .MATRIX_N    (MATRIX_N)
    ) sys_module_i (
        .core_clk                            (core_clk),
        .resetn                              (resetn),

        .pulse_systolic_module               (pulse_systolic_module),

        .sys_module_forward_in_valid         (sys_module_forward_high_valid  [sys_module]),
        .sys_module_forward_in               (sys_module_forward_high        [sys_module]),
        .sys_module_forward_in_pass          (), // not used

        .sys_module_down_in_valid            (sys_module_down_in_high_valid  [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        .sys_module_down_in                  (sys_module_down_in_high        [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),

        .sys_module_forward_out_valid        (sys_module_forward_high_valid  [sys_module+1]),
        .sys_module_forward_out              (sys_module_forward_high        [sys_module+1]),
        .sys_module_forward_out_pass          (), // not used

        .sys_module_down_out_valid           (sys_module_down_out_high_valid [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        .sys_module_down_out                 (sys_module_down_out_high       [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        
        .bias_valid                          (bias_valid),
        .bias                                (sys_bias_per_row_high             [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        
        .activation_valid                    (activation_valid),
        .activation                          (layer_config_activation_function_value),

        .shift_valid                         (shift_sys_module),

        .sys_module_pe_acc                   (sys_module_pe_acc_high   [sys_module]),

        .diagonal_flush_done                 (sys_module_flush_done    [sys_module]),

        .layer_config_leaky_relu_alpha_value ('1), // not used

        .debug_update_counter                (debug_update_counter_high     [sys_module])
    );
    fixed_cast_single #(
        .IN_WIDTH (HIGH_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*HIGH_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_high_row_0_i (
        .data_in    (sys_module_pe_acc_high[sys_module][0][0]),
        .data_out   (sys_module_pe_acc_casted[sys_module][0])
    );
    fixed_cast_single #(
        .IN_WIDTH (HIGH_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*HIGH_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_high_row_1_i (
        .data_in    (sys_module_pe_acc_high[sys_module][0][1]),
        .data_out   (sys_module_pe_acc_casted[sys_module][1])
    );
    fixed_cast_single #(
        .IN_WIDTH (HIGH_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*HIGH_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_high_row_2_i (
        .data_in    (sys_module_pe_acc_high[sys_module][0][2]),
        .data_out   (sys_module_pe_acc_casted[sys_module][2])
    );
    fixed_cast_single #(
        .IN_WIDTH (HIGH_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*HIGH_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_high_row_3_i (
        .data_in    (sys_module_pe_acc_high[sys_module][0][3]),
        .data_out   (sys_module_pe_acc_casted[sys_module][3])
    );
end

// Low precision systolic array
for (genvar sys_module = 0; sys_module < LOW_PRECISION_SYSTOLIC_MODULE_COUNT; sys_module++) begin
    // Driving from weight channel
    always_comb begin
        sys_module_down_in_low_valid [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] = {MATRIX_N{weight_channel_resp_valid}} & weight_channel_resp.valid_mask[(HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module)*MATRIX_N + (MATRIX_N-1) : (HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module)*MATRIX_N];
    end

    always_ff @(posedge core_clk or negedge resetn) begin
        if (!resetn | clean_sys_module_active) begin
            // sys_module_active_m [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module+1 : HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module] <= '0;
            sys_module_active_m [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module] <= '0;
        end else begin
            // sys_module_active_m [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module+1 : HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module] <= |sys_module_down_in_low_valid[sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] | sys_module_active_m [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT+sys_module+1 : HIGH_PRECISION_SYSTOLIC_MODULE_COUNT+sys_module];
            sys_module_active_m [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + sys_module] <= |sys_module_down_in_low_valid[sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] | sys_module_active_m [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT+sys_module];
        end
    end

    for (genvar index = sys_module*MATRIX_N; index < (sys_module*MATRIX_N + (MATRIX_N-1) + 1); index++) begin
        always_comb begin
            sys_module_down_in_low       [index] = weight_channel_resp.data[HIGH_PRECISION_SYSTOLIC_MODULE_COUNT*MATRIX_N+index][LOW_PRECISION_DATA_WIDTH-1:0];
            sys_bias_per_row_low         [index] = layer_config_bias_value[HIGH_PRECISION_SYSTOLIC_MODULE_COUNT*MATRIX_N+index][LOW_PRECISION_DATA_WIDTH-1:0];
        end
    end

    systolic_module #(
        .PRECISION (PRECISION),
        .FLOAT_WIDTH (FLOAT_WIDTH),
        .DATA_WIDTH  (LOW_PRECISION_DATA_WIDTH),
        .ACCUMULATOR_WIDTH (LOW_PRECISION_ACCUMULATOR_WIDTH),
        .FRACTIONAL_BITS (LOW_PRECISION_FRACTIONAL_BITS),
        .PASS_THROUGH_DATA_WIDTH(HIGH_PRECISION_DATA_WIDTH),
        .MATRIX_N    (MATRIX_N)
    ) sys_module_i (
        .core_clk                            (core_clk),
        .resetn                              (resetn),

        .pulse_systolic_module               (pulse_systolic_module),

        .sys_module_forward_in_valid         (sys_module_forward_low_valid  [sys_module]),
        .sys_module_forward_in               (sys_module_forward_low        [sys_module]),
        .sys_module_forward_in_pass          (sys_module_forward_high_pass  [sys_module]),

        .sys_module_down_in_valid            (sys_module_down_in_low_valid  [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        .sys_module_down_in                  (sys_module_down_in_low        [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),

        .sys_module_forward_out_valid        (sys_module_forward_low_valid  [sys_module+1]),
        .sys_module_forward_out              (sys_module_forward_low        [sys_module+1]),
        .sys_module_forward_out_pass         (sys_module_forward_high_pass  [sys_module+1]),

        .sys_module_down_out_valid           (sys_module_down_out_low_valid [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        .sys_module_down_out                 (sys_module_down_out_low       [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        
        .bias_valid                          (bias_valid),
        .bias                                (sys_bias_per_row_low      [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        
        .activation_valid                    (activation_valid),
        .activation                          (layer_config_activation_function_value),

        .shift_valid                         (shift_sys_module),

        .sys_module_pe_acc                   (sys_module_pe_acc_low    [sys_module]),

        .diagonal_flush_done                 (sys_module_flush_done    [HIGH_PRECISION_SYSTOLIC_MODULE_COUNT+sys_module]),

        .layer_config_leaky_relu_alpha_value ('1), // not used

        .debug_update_counter                (debug_update_counter_low     [sys_module])
    );

    fixed_cast_single #(
        .IN_WIDTH (LOW_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*LOW_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_low_row_0_i (
        .data_in    (sys_module_pe_acc_low[sys_module][0][0]),
        .data_out   (sys_module_pe_acc_casted[sys_module+HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][0])
    );
    fixed_cast_single #(
        .IN_WIDTH (LOW_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*LOW_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_low_row_1_i (
        .data_in    (sys_module_pe_acc_low[sys_module][0][1]),
        .data_out   (sys_module_pe_acc_casted[sys_module+HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][1])
    );
    fixed_cast_single #(
        .IN_WIDTH (LOW_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*LOW_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_low_row_2_i (
        .data_in    (sys_module_pe_acc_low[sys_module][0][2]),
        .data_out   (sys_module_pe_acc_casted[sys_module+HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][2])
    );
    fixed_cast_single #(
        .IN_WIDTH (LOW_PRECISION_ACCUMULATOR_WIDTH),
        .IN_FRAC_WIDTH (2*LOW_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS)
    ) casting_low_row_3_i (
        .data_in    (sys_module_pe_acc_low[sys_module][0][3]),
        .data_out   (sys_module_pe_acc_casted[sys_module+HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][3])
    );
end

// ==================================================================================================================================================
// Casting unit between two systolic arrays
// ==================================================================================================================================================
for (genvar casting_row = 0; casting_row < MATRIX_N; casting_row++) begin
    fixed_cast_single #(
        .IN_WIDTH (HIGH_PRECISION_DATA_WIDTH),
        .IN_FRAC_WIDTH (HIGH_PRECISION_FRACTIONAL_BITS),
        .OUT_WIDTH (LOW_PRECISION_DATA_WIDTH),
        .OUT_FRAC_WIDTH (LOW_PRECISION_FRACTIONAL_BITS)
    ) casting_row_i (
        .data_in    (sys_module_forward_high[HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][casting_row]),
        .data_out   (sys_module_forward_low[0][casting_row])
    );
    always_comb begin
        sys_module_forward_low_valid[0][casting_row] = sys_module_forward_high_valid[HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][casting_row];
        sys_module_forward_high_pass[0][casting_row] = sys_module_forward_high[HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][casting_row];
    end
end

assign mp_sys_module_forward_out_high_valid = sys_module_forward_low_valid[LOW_PRECISION_SYSTOLIC_MODULE_COUNT];
assign mp_sys_module_forward_out_high = sys_module_forward_high_pass[LOW_PRECISION_SYSTOLIC_MODULE_COUNT];

endmodule