
import top_pkg::*;

module feature_transformation_core #(
    parameter PRECISION = top_pkg::FLOAT_32,
    parameter FLOAT_WIDTH = 32,
    parameter DATA_WIDTH = 32,

    parameter MATRIX_N = top_pkg::TRANSFORMATION_ROWS,
    parameter SYSTOLIC_MODULE_COUNT = 32
) (
    input logic                                                                                       core_clk,
    input logic                                                                                       resetn,

    // Node Scoreboard -> Transformation Engine Interface
    input  logic                                                                                      nsb_fte_req_valid,
    output logic                                                                                      nsb_fte_req_ready,
    input  NSB_FTE_REQ_t                                                                              nsb_fte_req,

    output logic                                                                                      nsb_fte_resp_valid,
    input  logic                                                                                      nsb_fte_resp_ready,
    output NSB_FTE_RESP_t                                                                             nsb_fte_resp,

    // // Aggregation Buffer Interface
    // input  logic [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0] [top_pkg::NODE_ID_WIDTH-1:0]                 transformation_core_aggregation_buffer_node_id,
    // output logic [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0]                                              transformation_core_aggregation_buffer_pop,
    // input  logic [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0]                                              transformation_core_aggregation_buffer_out_feature_valid,
    // input  logic [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0] [top_pkg::AGGREGATION_BUFFER_READ_WIDTH-1:0] transformation_core_aggregation_buffer_out_feature,
    // input  logic [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0]                                              transformation_core_aggregation_buffer_slot_free,

    // Weight Channels: FTE -> Prefetcher Weight Bank (REQ)
    output logic                                                                                      weight_channel_req_valid,
    input  logic                                                                                      weight_channel_req_ready,
    output WEIGHT_CHANNEL_REQ_t                                                                       weight_channel_req,

    input  logic                                                                                      weight_channel_resp_valid,
    output logic                                                                                      weight_channel_resp_ready,
    input  WEIGHT_CHANNEL_RESP_t                                                                      weight_channel_resp,

    // Feature Channels: FTE -> Prefetcher Feature Bank (REQ)
    output logic                                                                                      feature_channel_req_valid,
    input  logic                                                                                      feature_channel_req_ready,
    output WEIGHT_CHANNEL_REQ_t                                                                       feature_channel_req,
                                                                
    input  logic                                                                                      feature_channel_resp_valid,
    output logic                                                                                      feature_channel_resp_ready,
    input  WEIGHT_CHANNEL_RESP_t                                                                      feature_channel_resp,

    // AXI Write Master Interface
    output logic                                                                                      axi_write_master_req_valid,
    input  logic                                                                                      axi_write_master_req_ready,
    output logic [33:0]                                                                               axi_write_master_req_start_address,
    output logic [7:0]                                                                                axi_write_master_req_len,

    input  logic                                                                                      axi_write_master_pop,
    output logic                                                                                      axi_write_master_data_valid,
    output logic [511:0]                                                                              axi_write_master_data,

    input  logic                                                                                      axi_write_master_resp_valid,
    output logic                                                                                      axi_write_master_resp_ready

    // Layer configuration
    // input  logic [9:0]                                                                                layer_config_in_features_count,
    // input  logic [9:0]                                                                                layer_config_out_features_count,  # the number of feature in a row of the systolic array output
    // input  logic [1:0]                                                                                layer_config_out_features_address_msb_value,
    // input  logic [31:0]                                                                               layer_config_out_features_address_lsb_value,
    // input  logic [31:0]                                                                               layer_config_bias_value,
    // input  logic [1:0]                                                                                layer_config_activation_function_value,
    // input  logic [31:0]                                                                               layer_config_leaky_relu_alpha_value,
    // input  logic [0:0]                                                                                ctrl_buffering_enable_value,
    // input  logic [0:0]                                                                                ctrl_writeback_enable_value
);

parameter SYS_MODULES_PER_BEAT = 512 / (MATRIX_N * FLOAT_WIDTH);
parameter MAX_WRITEBACK_BEATS_PER_NODESLOT = SYSTOLIC_MODULE_COUNT / SYS_MODULES_PER_BEAT;

typedef enum logic [3:0] {
    FTE_FSM_IDLE, FTE_FSM_REQ, FTE_FSM_REQ_WC, FTE_FSM_REQ_FC, FTE_FSM_MULT_SLOW, FTE_FSM_MULT_FAST, FTE_FSM_BIAS, FTE_FSM_ACTIVATION, FTE_FSM_BUFFER, FTE_FSM_WRITEBACK_REQ, FTE_FSM_WRITEBACK_RESP, FTE_FSM_SHIFT, FTE_FSM_NSB_RESP
} FTE_FSM_e;


// ==================================================================================================================================================
// Write Back Logic
// ==================================================================================================================================================
// Layer configuration
logic [9:0]                                                                                layer_config_out_channel_count;
logic [9:0]                                                                                layer_config_out_features_count;
logic [1:0]                                                                                layer_config_out_features_address_msb_value;
logic [31:0]                                                                               layer_config_out_features_address_lsb_value;
logic [31:0]                                                                               layer_config_bias_value;
logic [1:0]                                                                                layer_config_activation_function_value;
logic [31:0]                                                                               layer_config_leaky_relu_alpha_value;
logic [0:0]                                                                                ctrl_buffering_enable_value;
logic [0:0]                                                                                ctrl_writeback_enable_value;

assign layer_config_out_features_count =  10'd16; // top_pkg::MAX_FEATURE_COUNT;
assign layer_config_out_channel_count = 10'd4; // e.g. If output matrix is 4 X 8, the number of channel is 4. 
assign layer_config_out_features_address_msb_value = 2'b10;
assign layer_config_out_features_address_lsb_value = 32'd0; 
assign layer_config_bias_value = 32'd0;                     // no bias
assign layer_config_activation_function_value = 2'b00;      // no activation
assign layer_config_leaky_relu_alpha_value = 32'd0;
assign ctrl_buffering_enable_value = 1'b0;                  // no buffering (We don't need this)
assign ctrl_writeback_enable_value = 1'b1;

// ==================================================================================================================================================
// Declarations
// ==================================================================================================================================================

FTE_FSM_e fte_state, fte_state_n;
logic last_weight_resp_received, last_feature_resp_received;

// NSB requests
// logic [top_pkg::MAX_NODESLOT_COUNT-1:0]         nsb_req_nodeslots_q;
// logic [$clog2(top_pkg::MAX_NODESLOT_COUNT)-1:0] nodeslot_count;
// logic [$clog2(top_pkg::MAX_NODESLOT_COUNT)-1:0] nodeslots_to_buffer;
// logic [$clog2(top_pkg::MAX_NODESLOT_COUNT)-1:0] nodeslots_to_writeback;
logic [$clog2(top_pkg::MAX_NODESLOT_COUNT)-1:0] output_row_to_writeback;
logic [$clog2(top_pkg::MAX_NODESLOT_COUNT)-1:0] total_row_to_writeback;

// Systolic modules
// -------------------------------------------------------------------------------------

// Driven from aggregation buffer
logic [SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0]                                  sys_module_forward_valid; // 16
logic [SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0] [DATA_WIDTH-1:0]                 sys_module_forward;

// Driven from weight channel
logic [MAX_FEATURE_COUNT-1:0]                                                   sys_module_down_in_valid;
logic [MAX_FEATURE_COUNT-1:0] [DATA_WIDTH-1:0]                                  sys_module_down_in;

logic [MAX_FEATURE_COUNT-1:0]                                                   sys_module_down_out_valid;
logic [MAX_FEATURE_COUNT-1:0] [DATA_WIDTH-1:0]                                  sys_module_down_out;

logic [SYSTOLIC_MODULE_COUNT-1:0]                                               sys_module_flush_done;

logic [SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N:0] [MATRIX_N-1:0] [DATA_WIDTH-1:0]  sys_module_pe_acc;

logic                                                                           shift_sys_module;
logic                                                                           bias_valid;
logic                                                                           activation_valid;


// Driving systolic modules
// -------------------------------------------------------------------------------------

logic                                                 begin_feature_dump;
logic                                                 pulse_systolic_module;
// logic [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0]         slot_pop_shift;
// logic [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0]         busy_aggregation_slots_snapshot;
// logic                                                 pe_delay_counter;

// // Flushing logic
// logic [top_pkg::TRANSFORMATION_BUFFER_SLOTS-1:0]     transformation_buffer_slot_arb_oh;

// Writeback logic
logic [$clog2(MAX_WRITEBACK_BEATS_PER_NODESLOT):0]   sent_writeback_beats;
logic [$clog2(MAX_WRITEBACK_BEATS_PER_NODESLOT):0]   writeback_required_beats;
// logic [MATRIX_N:0] [top_pkg::NODE_ID_WIDTH-1:0]      sys_module_node_id_snapshot;
logic [$clog2(top_pkg::MAX_FEATURE_COUNT * 4) - 1:0] out_features_required_bytes;

logic [$clog2(MATRIX_N-1)-1:0]                       fast_pulse_counter;

logic [SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [MATRIX_N-1:0] [31:0] debug_update_counter;
logic [SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [MATRIX_N-1:0] [31:0] debug_update_counter_inv;

assign debug_update_counter_inv = ~debug_update_counter;

logic bias_applied, activation_applied;

// ==================================================================================================================================================
// Instances
// ==================================================================================================================================================
for (genvar sys_module = 0; sys_module < SYSTOLIC_MODULE_COUNT; sys_module++) begin
    // Driving from weight channel
    always_comb begin
        sys_module_down_in_valid [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] = {MATRIX_N{weight_channel_resp_valid}} & weight_channel_resp.valid_mask[sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N];
        // sys_module_down_in       [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N] = weight_channel_resp.data[sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N];
    end

    for (genvar index = sys_module*MATRIX_N; index < (sys_module*MATRIX_N + (MATRIX_N-1) + 1); index++) begin
        always_comb begin
            sys_module_down_in       [index] = weight_channel_resp.data[index][DATA_WIDTH-1:0];
        end
    end

    // Driving from feature channel
    // logic [SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0]                                  sys_module_forward_valid;
    // logic [SYSTOLIC_MODULE_COUNT:0] [MATRIX_N-1:0] [DATA_WIDTH-1:0]                 sys_module_forward;
    
    systolic_module #(
        .PRECISION (PRECISION),
        .FLOAT_WIDTH (FLOAT_WIDTH),
        .DATA_WIDTH  (DATA_WIDTH),
        .MATRIX_N    (MATRIX_N)
    ) sys_module_i (
        .core_clk                            (core_clk),
        .resetn                              (resetn),

        .pulse_systolic_module               (pulse_systolic_module),

        .sys_module_forward_in_valid         (sys_module_forward_valid  [sys_module]),
        .sys_module_forward_in               (sys_module_forward        [sys_module]),

        .sys_module_down_in_valid            (sys_module_down_in_valid  [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        .sys_module_down_in                  (sys_module_down_in        [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),

        .sys_module_forward_out_valid        (sys_module_forward_valid  [sys_module+1]),
        .sys_module_forward_out              (sys_module_forward        [sys_module+1]),

        .sys_module_down_out_valid           (sys_module_down_out_valid [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        .sys_module_down_out                 (sys_module_down_out       [sys_module*MATRIX_N + (MATRIX_N-1) : sys_module*MATRIX_N]),
        
        .bias_valid                          (bias_valid),
        .bias                                (layer_config_bias_value [DATA_WIDTH-1:0]),
        
        .activation_valid                    (activation_valid),
        .activation                          (layer_config_activation_function_value),

        .shift_valid                         (shift_sys_module),

        .sys_module_pe_acc                   (sys_module_pe_acc        [sys_module]),

        .diagonal_flush_done                 (sys_module_flush_done    [sys_module]),

        .layer_config_leaky_relu_alpha_value (layer_config_leaky_relu_alpha_value [DATA_WIDTH-1:0]),

        .debug_update_counter                (debug_update_counter     [sys_module])
    );

end

// // Transformation Buffer slot arbitration
// // --------------------------------------------------------------------------------

// rr_arbiter #(
//     .NUM_REQUESTERS     (top_pkg::TRANSFORMATION_BUFFER_SLOTS)
// ) prefetcher_req_arb (
//     .clk                (core_clk),
//     .resetn             (resetn),
//     .request            (transformation_buffer_slot_free),

//     // update when starting to flush new row
//     .update_lru         (fte_state == FTE_FSM_BUFFER),
//     .grant_oh           (transformation_buffer_slot_arb_oh)
// );

// // Driving Aggregation Buffer
// // --------------------------------------------------------------------------------

// hybrid_buffer_driver #(
//     .BUFFER_SLOTS (top_pkg::AGGREGATION_BUFFER_SLOTS),
//     .MAX_PULSES_PER_SLOT (top_pkg::MAX_FEATURE_COUNT)
// ) aggregation_buffer_driver_i (
//     .core_clk,
//     .resetn,

//     .begin_dump     (begin_feature_dump), // start dumping features from aggregation buffer
//     .pulse          (pulse_systolic_module), // pulse when systolic module is ready to accept new features

//     .pulse_limit    (layer_config_in_features_count), // number of features to dump

//     .slot_pop_shift (slot_pop_shift) // pop slot from aggregation buffer
// );

// ==================================================================================================================================================
// Logic
// ==================================================================================================================================================

// FTE State Machine
// -------------------------------------------------------------------------------------

always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        fte_state <= FTE_FSM_IDLE;
    end else begin
        fte_state <= fte_state_n;
    end
end

always_comb begin
    fte_state_n = fte_state;

    case(fte_state)

        FTE_FSM_IDLE: begin
            fte_state_n = nsb_fte_req_valid ? FTE_FSM_REQ : FTE_FSM_IDLE;
        end

        FTE_FSM_REQ: begin
            fte_state_n = weight_channel_req_ready && feature_channel_req_ready ? FTE_FSM_MULT_SLOW 
                        : weight_channel_req_ready ? FTE_FSM_REQ_FC
                        : feature_channel_req_ready ? FTE_FSM_REQ_WC
                        : FTE_FSM_REQ;
        end

        FTE_FSM_REQ_WC: begin
            fte_state_n = weight_channel_req_ready ? FTE_FSM_MULT_SLOW : FTE_FSM_REQ_WC;
        end

        FTE_FSM_REQ_FC: begin
            fte_state_n = feature_channel_req_ready ? FTE_FSM_MULT_SLOW : FTE_FSM_REQ_FC;
        end

        FTE_FSM_MULT_SLOW: begin
            fte_state_n = last_weight_resp_received ? FTE_FSM_MULT_FAST : FTE_FSM_MULT_SLOW;
        end

        FTE_FSM_MULT_FAST: begin
            fte_state_n = fast_pulse_counter == MATRIX_N - 1 ? FTE_FSM_BIAS : FTE_FSM_MULT_FAST;
        end

        FTE_FSM_BIAS: begin
            fte_state_n = bias_applied ? FTE_FSM_ACTIVATION : FTE_FSM_BIAS;
        end

        FTE_FSM_ACTIVATION: begin
            fte_state_n = activation_applied && ctrl_buffering_enable_value ? FTE_FSM_BUFFER
                        : activation_applied ? FTE_FSM_WRITEBACK_REQ
                        : FTE_FSM_ACTIVATION;
        end
        
        FTE_FSM_BUFFER: begin
            fte_state_n = FTE_FSM_WRITEBACK_REQ; // buffering takes a single cycle
        end

        FTE_FSM_WRITEBACK_REQ: begin
            fte_state_n = axi_write_master_req_ready ? FTE_FSM_WRITEBACK_RESP : FTE_FSM_WRITEBACK_REQ;
        end

        FTE_FSM_WRITEBACK_RESP: begin
            fte_state_n =
                        // Sending last beat for last nodeslot
                        (output_row_to_writeback == 'd1) && (sent_writeback_beats == writeback_required_beats) && axi_write_master_resp_valid ? FTE_FSM_NSB_RESP
                        // Sending last beat, more nodeslots to go
                        : (sent_writeback_beats == writeback_required_beats) && axi_write_master_resp_valid ? FTE_FSM_SHIFT
                        : FTE_FSM_WRITEBACK_RESP;
        end

        FTE_FSM_SHIFT: begin
            fte_state_n = ctrl_buffering_enable_value ? FTE_FSM_BUFFER
                        : ctrl_writeback_enable_value ? FTE_FSM_WRITEBACK_REQ
                        : FTE_FSM_NSB_RESP;
        end

        FTE_FSM_NSB_RESP: begin
            fte_state_n = nsb_fte_resp_ready ? FTE_FSM_IDLE : FTE_FSM_NSB_RESP;
        end

        default: begin
            fte_state_n = FTE_FSM_IDLE;
        end

    endcase
end

// Bias and activation applied flag
always_ff @(posedge core_clk or negedge resetn) begin

    if (!resetn) begin
        bias_applied <= '0;
        activation_applied <= '0;

    end else begin
        bias_applied <= (fte_state == FTE_FSM_MULT_FAST && fte_state_n == FTE_FSM_BIAS) ? '0
                    : fte_state == FTE_FSM_BIAS ? '1
                    : bias_applied;

        activation_applied <= fte_state == FTE_FSM_BIAS && fte_state_n == FTE_FSM_ACTIVATION ? '0
                            : fte_state == FTE_FSM_ACTIVATION ? '1
                            : activation_applied;

    end
end

// // Take snapshot of busy slots and respective node IDs after NSB request
// // -------------------------------------------------------------------------------------

// always_ff @(posedge core_clk or negedge resetn) begin
//     if (!resetn) begin
//         busy_aggregation_slots_snapshot <= '0;
//         nsb_req_nodeslots_q <= '0;
        
//     // Starting multiplication
//     end else if ((fte_state == FTE_FSM_IDLE) && (fte_state_n == FTE_FSM_REQ_WC)) begin
//         busy_aggregation_slots_snapshot <= ~transformation_core_aggregation_buffer_slot_free;
//         nsb_req_nodeslots_q <= nsb_fte_req.nodeslots;
//     end
// end

// for (genvar row = 0; row < MATRIX_N; row++) begin
//     always_ff @(posedge core_clk or negedge resetn) begin
//         if (!resetn) begin
//             sys_module_node_id_snapshot [row]          <= '0;
            
//         // Starting multiplication
//         end else if ((fte_state == FTE_FSM_IDLE) && (fte_state_n == FTE_FSM_REQ_WC)) begin
//             sys_module_node_id_snapshot [row]          <= transformation_core_aggregation_buffer_node_id [row];
        
//         // Shift node IDs
//         end else if (fte_state == FTE_FSM_SHIFT) begin
//             sys_module_node_id_snapshot [row]          <= sys_module_node_id_snapshot [row+1];
//         end
//     end
// end

// assign sys_module_node_id_snapshot [MATRIX_N] = '0;

// Driving systolic module
// -------------------------------------------------------------------------------------

always_comb begin
    // Begin feature dump when weight channel request accepted
    begin_feature_dump = ((fte_state == FTE_FSM_REQ) || (fte_state == FTE_FSM_REQ_WC) || (fte_state == FTE_FSM_REQ_FC)) && (fte_state_n == FTE_FSM_MULT_SLOW);

    // Pulse module when features ready in aggregation buffer and weights ready in weight channel
    // pulse_systolic_module = ((fte_state_n == FTE_FSM_MULT_SLOW) && &transformation_core_aggregation_buffer_out_feature_valid && weight_channel_resp_valid) || fte_state == FTE_FSM_MULT_FAST;
    pulse_systolic_module = ((fte_state_n == FTE_FSM_MULT_SLOW) && (weight_channel_resp_valid || feature_channel_resp_valid)) || fte_state == FTE_FSM_MULT_FAST; // && feature_channel_resp_valid

    // TO DO: we need to change here too
    // Drive systolic module from aggregation buffer (on the left)
    // sys_module_forward_valid [0] = slot_pop_shift & busy_aggregation_slots_snapshot; // [MATRIX-1:0] = [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0]
    // sys_module_forward       [0] = transformation_core_aggregation_buffer_out_feature; // [MATRIX_N-1:0] [DATA_WIDTH-1:0] = [top_pkg::AGGREGATION_BUFFER_SLOTS-1:0] [top_pkg::AGGREGATION_BUFFER_READ_WIDTH-1:0]

    // transformation_core_aggregation_buffer_pop = slot_pop_shift & {MATRIX_N{pulse_systolic_module}} & busy_aggregation_slots_snapshot & ~transformation_core_aggregation_buffer_slot_free;
end

// Feature channel
// -------------------------------------------------------------------------------------
always_comb begin
    sys_module_forward_valid [0] = {MATRIX_N{feature_channel_resp_valid}} & feature_channel_resp.valid_mask[MATRIX_N-1:0];
end
for (genvar index = 0; index < MATRIX_N; index++) begin
    always_comb begin
        sys_module_forward       [0][index] = feature_channel_resp.data[index][DATA_WIDTH-1:0];
    end
end

// After SLOW phase done, pulse every cycle until last element propagates down to the end
always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        fast_pulse_counter <= '0;

    end else if (fte_state == FTE_FSM_MULT_SLOW && fte_state_n == FTE_FSM_MULT_FAST) begin
        fast_pulse_counter <= '0;

    end else if (fte_state == FTE_FSM_MULT_FAST) begin
        fast_pulse_counter <= fast_pulse_counter + 1'b1;
    end
end
    
always_comb begin
    // Bias and activation after multiplication finished
    bias_valid       = (fte_state == FTE_FSM_BIAS);
    activation_valid = (fte_state == FTE_FSM_ACTIVATION);
    shift_sys_module = (fte_state == FTE_FSM_SHIFT);
end

// // Buffering Logic
// // -------------------------------------------------------------------------------------

// for (genvar slot = 0; slot < TRANSFORMATION_BUFFER_SLOTS; slot++) begin
//     always_comb begin
//         transformation_buffer_write_enable  [slot] = transformation_buffer_slot_arb_oh[slot] && (fte_state == FTE_FSM_BUFFER);
//         transformation_buffer_write_address [slot] = '0;
//         transformation_buffer_write_data    [slot] = sys_module_pe_acc [0] [0]; // 16*32 bits = 512b
//     end    
// end

// always_ff @(posedge core_clk or negedge resetn) begin
//     if (!resetn) begin
//         nodeslots_to_buffer <= '0;
        
//     // Accepting NSB request
//     end else if (nsb_fte_req_valid && nsb_fte_req_ready) begin
//         nodeslots_to_buffer    <= ctrl_buffering_enable_value ? nodeslot_count : nodeslots_to_buffer;
    
//     // Done flushing a row of features
//     end else if (fte_state == FTE_FSM_BUFFER) begin
//         nodeslots_to_buffer <= nodeslots_to_buffer - 1'b1;
//     end
// end

// count_ones #(
//     .INPUT_WIDTH (top_pkg::MAX_NODESLOT_COUNT)
// ) count_nodeslots (
//     .data (nsb_fte_req.nodeslots),
//     .count (nodeslot_count)
// );

// Writeback Logic
// -------------------------------------------------------------------------------------

always_comb begin
    out_features_required_bytes = layer_config_out_features_count * 4; // 4 bytes per feature
    out_features_required_bytes = {out_features_required_bytes[$clog2(top_pkg::MAX_FEATURE_COUNT * 4) - 1 : 6], 6'd0} + (out_features_required_bytes[5:0] ? 'd64 : 1'b0); // nearest multiple of 64
    // Div feautre count by 16, round up
    writeback_required_beats = (layer_config_out_features_count >> 4) + (layer_config_out_features_count[3:0] ? 1'b1 : 1'b0);

    // TO DO: we need to change here too sys_module_node_id_snapshot
    // Request
    axi_write_master_req_valid = (fte_state == FTE_FSM_WRITEBACK_REQ);
    axi_write_master_req_start_address = {layer_config_out_features_address_msb_value, layer_config_out_features_address_lsb_value}
                                            + (total_row_to_writeback-output_row_to_writeback) * out_features_required_bytes;

    axi_write_master_req_len = writeback_required_beats - 1'b1;

    // Data
    axi_write_master_data_valid = (fte_state == FTE_FSM_WRITEBACK_RESP);
    axi_write_master_data = {sys_module_pe_acc [SYS_MODULES_PER_BEAT*sent_writeback_beats + 'd3][0],
                            sys_module_pe_acc  [SYS_MODULES_PER_BEAT*sent_writeback_beats + 'd2][0],
                            sys_module_pe_acc  [SYS_MODULES_PER_BEAT*sent_writeback_beats + 'd1][0],
                            sys_module_pe_acc  [SYS_MODULES_PER_BEAT*sent_writeback_beats + 'd0][0]
                        };

    // logic [SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N:0] [MATRIX_N-1:0] [DATA_WIDTH-1:0]  sys_module_pe_acc;
    // Response
    axi_write_master_resp_ready = (fte_state == FTE_FSM_WRITEBACK_RESP);
end


always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        output_row_to_writeback <= '0;
        sent_writeback_beats <= '0;

    // Accepting NSB request
    end else if (nsb_fte_req_valid && nsb_fte_req_ready) begin
        total_row_to_writeback <= ctrl_writeback_enable_value ? layer_config_out_channel_count : output_row_to_writeback; // TODO: hardcored to 4
        output_row_to_writeback <= ctrl_writeback_enable_value ? layer_config_out_channel_count : output_row_to_writeback; // TODO: hardcored to 4
        sent_writeback_beats <= '0;

    // Accepting AXI Write Master request
    end else if (axi_write_master_req_valid && axi_write_master_req_ready) begin
        sent_writeback_beats <= '0;

    // Sent 512b beat
    end else if (axi_write_master_pop) begin
        sent_writeback_beats <= sent_writeback_beats + 1'b1;

    // Accepting write response
    end else if (fte_state == FTE_FSM_WRITEBACK_RESP && axi_write_master_resp_valid && sent_writeback_beats == writeback_required_beats) begin
        output_row_to_writeback <= output_row_to_writeback - 1'b1;
    end
end

// NSB Interface
// -------------------------------------------------------------------------------------

always_comb begin
    nsb_fte_req_ready           = (fte_state == FTE_FSM_IDLE);
    
    // TO DO: NSB resp
    nsb_fte_resp_valid          = (fte_state == FTE_FSM_NSB_RESP);
    // nsb_fte_resp.nodeslots      = nsb_req_nodeslots_q;
end

// Weight Channel Interface
// -------------------------------------------------------------------------------------

always_comb begin
    weight_channel_req_valid  = (fte_state == FTE_FSM_REQ) || (fte_state == FTE_FSM_REQ_WC);

    // Feature counts aren't used by weight bank
    weight_channel_req.in_features  = top_pkg::MAX_FEATURE_COUNT;
    weight_channel_req.out_features = top_pkg::MAX_FEATURE_COUNT;

    // Accept weight bank response when pulsing systolic module (i.e. aggregation buffer is also ready)
    weight_channel_resp_ready = (fte_state == FTE_FSM_MULT_SLOW) && (pulse_systolic_module || weight_channel_resp.done);
end

// Feature Channel Interface
// -------------------------------------------------------------------------------------

always_comb begin
    feature_channel_req_valid  = (fte_state == FTE_FSM_REQ) || (fte_state == FTE_FSM_REQ_FC);

    // Feature counts aren't used by weight bank
    feature_channel_req.in_features  = top_pkg::MAX_FEATURE_COUNT;
    feature_channel_req.out_features = top_pkg::MAX_FEATURE_COUNT;

    // Accept weight bank response when pulsing systolic module (i.e. aggregation buffer is also ready)
    feature_channel_resp_ready = (fte_state == FTE_FSM_MULT_SLOW) && (pulse_systolic_module || feature_channel_resp.done);
end

// Raise flag as pre-condition for transitioning from MULT state 
// TO DO: assume that the feature and weight channel responses the same way. So when the last weight resp received means the last feature resp received too.
always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        last_weight_resp_received <= '0;
        
    // Starting new request
    end else if ((fte_state == FTE_FSM_IDLE) && nsb_fte_req_valid) begin
        last_weight_resp_received <= '0;
    
    end else if (weight_channel_resp_valid && weight_channel_resp.done) begin
        last_weight_resp_received <= '1;
    end
end

endmodule