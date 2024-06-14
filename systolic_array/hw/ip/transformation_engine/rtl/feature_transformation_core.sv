
import top_pkg::*;

module feature_transformation_core #(
    parameter PRECISION = top_pkg::FLOAT_32,
    parameter FLOAT_WIDTH = 32,
    parameter DATA_WIDTH = 32,

    parameter AXI_ADDRESS_WIDTH = 32,
    parameter AXI_DATA_WIDTH = 512,

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
    output logic [AXI_ADDRESS_WIDTH-1:0]                                                              axi_write_master_req_start_address,
    output logic [7:0]                                                                                axi_write_master_req_len,

    input  logic                                                                                      axi_write_master_pop,
    output logic                                                                                      axi_write_master_data_valid,
    output logic [AXI_DATA_WIDTH-1:0]                                                                 axi_write_master_data,

    input  logic                                                                                      axi_write_master_resp_valid,
    output logic                                                                                      axi_write_master_resp_ready,

    // Layer configuration
    input  logic [31:0]                                                                               layer_config_out_channel_count,
    input  logic [31:0]                                                                               layer_config_out_features_count,  // the number of feature in a row of the systolic array output
    input  logic [1:0]                                                                                layer_config_out_features_address_msb_value,
    input  logic [AXI_ADDRESS_WIDTH-2:0]                                                              layer_config_out_features_address_lsb_value,
    input  logic [CORE_COUNT*SYSTOLIC_MODULE_COUNT*MATRIX_N-1:0] [31:0]                               layer_config_bias_value,
    input  logic [1:0]                                                                                layer_config_activation_function_value,
    input  logic [31:0]                                                                               layer_config_leaky_relu_alpha_value,
    input  logic [0:0]                                                                                ctrl_buffering_enable_value,
    input  logic [0:0]                                                                                ctrl_writeback_enable_value,

    // Write Back Configuration
    // input  logic [$clog2(MAX_WRITEBACK_BEATS_PER_NODESLOT):0]                                         writeback_required_beats
    input  logic [AXI_ADDRESS_WIDTH-1:0]                                                              writeback_offset             // offset is used to indicate the current block to write back 

);

parameter SYS_MODULES_PER_BEAT = 512 / (MATRIX_N * FLOAT_WIDTH);
parameter MAX_WRITEBACK_BEATS_PER_NODESLOT = (CORE_COUNT*SYSTOLIC_MODULE_COUNT) / SYS_MODULES_PER_BEAT; // The max beats require to write back all systolic modules

typedef enum logic [3:0] {
    FTE_FSM_IDLE, FTE_FSM_REQ, FTE_FSM_REQ_WC, FTE_FSM_REQ_FC, FTE_FSM_MULT_SLOW, FTE_FSM_MULT_FAST, FTE_FSM_BIAS, FTE_FSM_ACTIVATION, FTE_FSM_BUFFER, FTE_FSM_WRITEBACK_REQ, FTE_FSM_WRITEBACK_RESP, FTE_FSM_SHIFT, FTE_FSM_NSB_RESP
} FTE_FSM_e;


// ==================================================================================================================================================
// Write Back Logic
// ==================================================================================================================================================
// Layer configuration
// assign layer_config_out_features_count =  10'd16; // top_pkg::MAX_FEATURE_COUNT;
// assign layer_config_out_channel_count = 10'd4; // e.g. If output matrix is 4 X 8, the number of channel is 4. 
// assign layer_config_out_features_address_msb_value = 2'b10;
// assign layer_config_out_features_address_lsb_value = 32'd0; 
// assign layer_config_bias_value = 32'd0;                     // no bias
// assign layer_config_activation_function_value = 2'b00;      // no activation
// assign layer_config_leaky_relu_alpha_value = 32'd0;
// assign ctrl_buffering_enable_value = 1'b0;                  // no buffering (We don't need this)
// assign ctrl_writeback_enable_value = 1'b1;

// ==================================================================================================================================================
// Declarations
// ==================================================================================================================================================

FTE_FSM_e fte_state, fte_state_n;
logic last_weight_resp_received, last_feature_resp_received;

logic [$clog2(top_pkg::MAX_NODESLOT_COUNT)-1:0] output_row_to_writeback;
logic [$clog2(top_pkg::MAX_NODESLOT_COUNT)-1:0] total_row_to_writeback;

// Systolic modules
// -------------------------------------------------------------------------------------
logic [CORE_COUNT:0] [MATRIX_N-1:0]                                   mp_sys_module_forward_high_valid; // 16
logic [CORE_COUNT:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]   mp_sys_module_forward_high; // input is always in high precision
// Driven from weight channel
// logic [CORE_COUNT*HIGH_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_pe_acc_high_casted; // NOTE: Data will get writeback in a high precision format. Low precision will get padded back to high precision by using 0.
// logic [CORE_COUNT*LOW_PRECISION_SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_pe_acc_low_casted; // NOTE: Data will get writeback in a high precision format. Low precision will get padded back to high precision by using 0.

logic [CORE_COUNT*SYSTOLIC_MODULE_COUNT-1:0] [MATRIX_N-1:0] [HIGH_PRECISION_DATA_WIDTH-1:0]  sys_module_pe_acc_casted; // NOTE: Data will get writeback in a high precision format. Low precision will get padded back to high precision by using 0.

logic                                                                           shift_sys_module;
logic                                                                           bias_valid;
logic                                                                           activation_valid;

WEIGHT_CHANNEL_RESP_t [CORE_COUNT-1:0]                                          mp_weight_channel_resp;
// Driving systolic modules
// -------------------------------------------------------------------------------------
logic                                                 begin_feature_dump;
logic                                                 pulse_systolic_module;


// Writeback logic
logic [$clog2(MAX_WRITEBACK_BEATS_PER_NODESLOT):0]   sent_writeback_beats;
logic [$clog2(top_pkg::MAX_FEATURE_COUNT/16):0]      writeback_required_beats_intermediate;
logic [$clog2(MAX_WRITEBACK_BEATS_PER_NODESLOT):0]   writeback_required_beats;
logic [$clog2(top_pkg::MAX_FEATURE_COUNT * 4) - 1:0] out_features_required_bytes;

logic [$clog2(MATRIX_N-1)-1:0]                       fast_pulse_counter;

logic bias_applied, activation_applied;
// Writeback AXI_DATA_WIDTH shifting logic (shifting horizontally)
logic [CORE_COUNT*SYSTOLIC_MODULE_COUNT-1:0] sys_module_active;                                      // use to recall if a systolic array is active or not to write out the result
logic [CORE_COUNT-1:0][$clog2(SYSTOLIC_MODULE_COUNT)-1:0] sys_module_active_count;
logic [SYS_MODULES_PER_BEAT:0] [AXI_DATA_WIDTH-1:0] axi_write_master_data_i;    // use to shift the data horizontally

// ==================================================================================================================================================
// Instances
// ==================================================================================================================================================
for (genvar mp_sys_module = 0; mp_sys_module < CORE_COUNT; mp_sys_module++) begin
    always_comb begin
        mp_weight_channel_resp[mp_sys_module].valid_mask = weight_channel_resp.valid_mask[(mp_sys_module*SYSTOLIC_MODULE_COUNT*MATRIX_N) + (SYSTOLIC_MODULE_COUNT*MATRIX_N-1) : mp_sys_module*SYSTOLIC_MODULE_COUNT*MATRIX_N];
        mp_weight_channel_resp[mp_sys_module].data = weight_channel_resp.data[mp_sys_module*SYSTOLIC_MODULE_COUNT*MATRIX_N + SYSTOLIC_MODULE_COUNT*MATRIX_N-1 : mp_sys_module*SYSTOLIC_MODULE_COUNT*MATRIX_N];
    end

    (* keep_hierarchy = "yes" *) mixed_precision_systolic_module #(
        .PRECISION(PRECISION),
        .FLOAT_WIDTH(FLOAT_WIDTH),
        .MATRIX_N(MATRIX_N),
        .SYSTOLIC_MODULE_COUNT(SYSTOLIC_MODULE_COUNT)
    ) mp_sys (
        .core_clk(core_clk),
        .resetn(resetn),

        .weight_channel_resp_valid                  (weight_channel_resp_valid),
        .weight_channel_resp                        (mp_weight_channel_resp             [mp_sys_module]),

        .mp_sys_module_forward_high_valid           (mp_sys_module_forward_high_valid   [mp_sys_module]),
        .mp_sys_module_forward_high                 (mp_sys_module_forward_high         [mp_sys_module]),
        .mp_sys_module_forward_out_high_valid       (mp_sys_module_forward_high_valid   [mp_sys_module+1]),
        .mp_sys_module_forward_out_high             (mp_sys_module_forward_high         [mp_sys_module+1]),

        // .sys_module_pe_acc_high_casted              (sys_module_pe_acc_high_casted      [mp_sys_module*HIGH_PRECISION_SYSTOLIC_MODULE_COUNT + HIGH_PRECISION_SYSTOLIC_MODULE_COUNT-1 : mp_sys_module*HIGH_PRECISION_SYSTOLIC_MODULE_COUNT]),
        // .sys_module_pe_acc_low_casted               (sys_module_pe_acc_low_casted       [mp_sys_module*LOW_PRECISION_SYSTOLIC_MODULE_COUNT + LOW_PRECISION_SYSTOLIC_MODULE_COUNT-1 : mp_sys_module*LOW_PRECISION_SYSTOLIC_MODULE_COUNT]),
        .sys_module_pe_acc_casted                   (sys_module_pe_acc_casted           [mp_sys_module*SYSTOLIC_MODULE_COUNT + SYSTOLIC_MODULE_COUNT-1 : mp_sys_module*SYSTOLIC_MODULE_COUNT]),

        .layer_config_bias_value                    (layer_config_bias_value            [mp_sys_module*SYSTOLIC_MODULE_COUNT*MATRIX_N + SYSTOLIC_MODULE_COUNT*MATRIX_N-1 : mp_sys_module*SYSTOLIC_MODULE_COUNT*MATRIX_N]),
        .layer_config_activation_function_value     (layer_config_activation_function_value),
        .layer_config_leaky_relu_alpha_value        (layer_config_leaky_relu_alpha_value),

        .pulse_systolic_module                      (pulse_systolic_module),
        .shift_sys_module                           (shift_sys_module),
        .bias_valid                                 (bias_valid),
        .activation_valid                           (activation_valid),

        // Writeback AXI_DATA_WIDTH shifting logic (shifting horizontally)
        .sys_module_active                          (sys_module_active                  [mp_sys_module*SYSTOLIC_MODULE_COUNT + SYSTOLIC_MODULE_COUNT-1 : mp_sys_module*SYSTOLIC_MODULE_COUNT]), // TODO: This is not implemented
        .sys_module_active_count                    (sys_module_active_count            [mp_sys_module]),
        .clean_sys_module_active                    (fte_state_n == FTE_FSM_IDLE)
    );

end
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

        FTE_FSM_NSB_RESP: begin // if the nsb response is ready, meaning we have finished the write back for the current nodeslot and shift one last time to clean the entire row
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

// Driving systolic module
// -------------------------------------------------------------------------------------

always_comb begin
    // Begin feature dump when weight channel request accepted
    begin_feature_dump = ((fte_state == FTE_FSM_REQ) || (fte_state == FTE_FSM_REQ_WC) || (fte_state == FTE_FSM_REQ_FC)) && (fte_state_n == FTE_FSM_MULT_SLOW);

    // Pulse module when features ready in aggregation buffer and weights ready in weight channel
    pulse_systolic_module = ((fte_state_n == FTE_FSM_MULT_SLOW) && (weight_channel_resp_valid || feature_channel_resp_valid)) || fte_state == FTE_FSM_MULT_FAST;
end

// Feature channel
// -------------------------------------------------------------------------------------
always_comb begin
    mp_sys_module_forward_high_valid [0] = {MATRIX_N{feature_channel_resp_valid}} & feature_channel_resp.valid_mask[MATRIX_N-1:0];
end
for (genvar index = 0; index < MATRIX_N; index++) begin
    always_comb begin
        mp_sys_module_forward_high       [0][index] = feature_channel_resp.data[index][HIGH_PRECISION_DATA_WIDTH-1:0];
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
    shift_sys_module = (fte_state == FTE_FSM_SHIFT) || (fte_state == FTE_FSM_NSB_RESP && fte_state_n == FTE_FSM_IDLE); // Added FTE_FSM_IDLE to shift the last row out which is equivalant to reseting the systolic array
end

// Writeback Logic
// -------------------------------------------------------------------------------------

always_comb begin
    out_features_required_bytes = layer_config_out_features_count * 4; // 4 bytes per feature
    out_features_required_bytes = {out_features_required_bytes[$clog2(top_pkg::MAX_FEATURE_COUNT * 4) - 1 : 6], 6'd0} + (out_features_required_bytes[5:0] ? 'd64 : 1'b0); // nearest multiple of 64, a row of the whole output matrix

    // Div feautre count by 16, round up
    writeback_required_beats_intermediate = (layer_config_out_features_count >> 4) + (layer_config_out_features_count[3:0] ? 1'b1 : 1'b0); // TODO: I think div by 16 is wrong here. One beat should be 64 bytes. Intermediate value with a larger width to capture the occurance of number greater than 32
    // writeback_required_beats_intermediate = (layer_config_out_features_count >> 6) + (layer_config_out_features_count[5:0] ? 1'b1 : 1'b0); // TODO: I think div by 16 is wrong here. One beat should be 64 bytes. Intermediate value with a larger width to capture the occurance of number greater than 32
    writeback_required_beats = (writeback_required_beats_intermediate >= MAX_WRITEBACK_BEATS_PER_NODESLOT)? MAX_WRITEBACK_BEATS_PER_NODESLOT: writeback_required_beats_intermediate; // TODO: For now this is hardcoded to be neight the feature count or systolic array count. Ideally this should be number of beats needed to be transfered
    // writeback_required_beats = (sys_module_active_count * MATRIX_N) >> 4 + ((sys_module_active_count * MATRIX_N)[3:0] ? 1'b1 : 1'b0); TODO: Use this
    // Request
    axi_write_master_req_valid = (fte_state == FTE_FSM_WRITEBACK_REQ);
    axi_write_master_req_start_address = {layer_config_out_features_address_msb_value, layer_config_out_features_address_lsb_value}
                                            + (total_row_to_writeback-output_row_to_writeback) * out_features_required_bytes + writeback_offset;
    axi_write_master_req_len = writeback_required_beats - 1'b1; // TODO: change writeback_required_beats to be the number of beats needed to be transfered
    // Data
    axi_write_master_data_valid = (fte_state == FTE_FSM_WRITEBACK_RESP);
    // Response
    axi_write_master_resp_ready = (fte_state == FTE_FSM_WRITEBACK_RESP);
end


for (genvar sys_module = 0; sys_module < 4; sys_module++) begin
    always_comb begin
        if(sys_module < SYSTOLIC_MODULE_COUNT & sys_module_active[(SYS_MODULES_PER_BEAT*(sent_writeback_beats)) + sys_module]) begin
            // if (((SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module) >= HIGH_PRECISION_SYSTOLIC_MODULE_COUNT) begin
            //     // one Low precision module
            //     axi_write_master_data_i[0][AXI_DATA_WIDTH-1-sys_module*MATRIX_N*32:AXI_DATA_WIDTH-1-((sys_module+1)*MATRIX_N*32-1)] = {
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_low_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module - HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][0],
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_low_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module - HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][1],
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_low_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module - HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][2],
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_low_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module - HIGH_PRECISION_SYSTOLIC_MODULE_COUNT][3]
            //     };
            // end else begin
            //     // one high precision module
            //     axi_write_master_data_i[0][AXI_DATA_WIDTH-1-sys_module*MATRIX_N*32:AXI_DATA_WIDTH-1-((sys_module+1)*MATRIX_N*32-1)] = {
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_high_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][0],
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_high_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][1],
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_high_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][2],
            //         {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_high_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][3]
            //     };
            // end
            axi_write_master_data_i[0][AXI_DATA_WIDTH-1-sys_module*MATRIX_N*32:AXI_DATA_WIDTH-1-((sys_module+1)*MATRIX_N*32-1)] = {
                {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][0],
                {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][1],
                {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][2],
                {{32 - HIGH_PRECISION_DATA_WIDTH} {1'd0}}, sys_module_pe_acc_casted[(SYS_MODULES_PER_BEAT*sent_writeback_beats)  + sys_module][3]
            };
        end else begin
            axi_write_master_data_i[0][AXI_DATA_WIDTH-1-sys_module*MATRIX_N*32:AXI_DATA_WIDTH-1-((sys_module+1)*MATRIX_N*32-1)] = 0;
        end
    end
end

// ripple shifter 
// TODO: As we are performing fix shifing here. Meaning the software has to assume the input feature is a multiple of 4
for (genvar sys_module = 0; sys_module < 4; sys_module++) begin
    always_comb begin
        // if(SYS_MODULES_PER_BEAT*sent_writeback_beats + sys_module > SYSTOLIC_MODULE_COUNT | !sys_module_active[SYS_MODULES_PER_BEAT*sent_writeback_beats + sys_module]) begin
        if(!sys_module_active[SYS_MODULES_PER_BEAT*sent_writeback_beats + sys_module]) begin
            axi_write_master_data_i[sys_module+1] = axi_write_master_data_i[sys_module] >> 128;
        end else begin
            axi_write_master_data_i[sys_module+1] = axi_write_master_data_i[sys_module];
        end
    end
end

always_comb begin
    axi_write_master_data = axi_write_master_data_i[4];
end


always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
        output_row_to_writeback <= '0;
        total_row_to_writeback <= '0;
        sent_writeback_beats <= '0;

    // Accepting NSB request
    end else if (nsb_fte_req_valid && nsb_fte_req_ready) begin
        total_row_to_writeback <= ctrl_writeback_enable_value ? TRANSFORMATION_ROWS : output_row_to_writeback; // TODO: I made a design decision to write back TRANSFORMATION_ROWS rows of the output matrix meaning the software has to ensure that the output channel is a multiple of 4. Empty rows will also get write back
        output_row_to_writeback <= ctrl_writeback_enable_value ? TRANSFORMATION_ROWS : output_row_to_writeback; // TODO: hardcored to TRANSFORMATION_ROWS currently
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