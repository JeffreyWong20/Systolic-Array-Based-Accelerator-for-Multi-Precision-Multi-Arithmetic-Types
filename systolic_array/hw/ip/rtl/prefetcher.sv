import top_pkg::*;
// import noc_pkg::*;

// module prefetcher #(
//     // parameter FETCH_TAG_COUNT = top_pkg::MESSAGE_CHANNEL_COUNT
// ) 
module prefetcher #
(
    parameter AXI_DATA_WIDTH = 512,
    parameter AXI_ADDR_WIDTH = 30,
    parameter MAX_FIFO_ROWS = (top_pkg::MAX_FEATURE_COUNT)
)
(
    input logic core_clk,
    input logic resetn,

    // input logic regbank_clk,
    // input logic regbank_resetn,

    // NSB -> Prefetcher Interface
    input  logic          nsb_prefetcher_req_valid,
    output logic          nsb_prefetcher_req_ready,
    input  NSB_PREF_REQ_t nsb_prefetcher_req,

    output logic           nsb_prefetcher_resp_valid,  // valid only for now
    output NSB_PREF_RESP_t nsb_prefetcher_resp,

    // // Register Bank
    // input  logic [AXIL_ADDR_WIDTH-1:0]        s_axi_awaddr,
    // input  logic [2:0]                        s_axi_awprot,
    // input  logic                              s_axi_awvalid,
    // output logic                              s_axi_awready,
    // input  logic [31:0]                       s_axi_wdata,
    // input  logic [3:0]                        s_axi_wstrb,
    // input  logic                              s_axi_wvalid,
    // output logic                              s_axi_wready,
    // input  logic [AXIL_ADDR_WIDTH-1:0]        s_axi_araddr,
    // input  logic [2:0]                        s_axi_arprot,
    // input  logic                              s_axi_arvalid,
    // output logic                              s_axi_arready,
    // output logic [31:0]                       s_axi_rdata,
    // output logic [1:0]                        s_axi_rresp,
    // output logic                              s_axi_rvalid,
    // input  logic                              s_axi_rready,
    // output logic [1:0]                        s_axi_bresp,
    // output logic                              s_axi_bvalid,
    // input  logic                              s_axi_bready,

    // Prefetcher Weight Bank Read Master -> AXI Memory Interconnect
    output logic [ AXI_ADDR_WIDTH-1:0]  prefetcher_weight_bank_rm_axi_interconnect_axi_araddr,
    output logic [  1:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arburst,
    output logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arcache,
    output logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arid,
    output logic [  7:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arlen,
    output logic [  0:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arlock,
    output logic [  2:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arprot,
    output logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arqos,
    output logic [  2:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_arsize,
    output logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid,
    input  logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_arready,
    output logic [ AXI_ADDR_WIDTH-1:0]  prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr,
    output logic [  1:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awburst,
    output logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awcache,
    output logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awid,
    output logic [  7:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awlen,
    output logic [  0:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awlock,
    output logic [  2:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awprot,
    output logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awqos,
    input  logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_awready,
    output logic [  2:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_awsize,
    output logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid,
    input  logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_bid,
    output logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_bready,
    input  logic [  1:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_bresp,
    input  logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid,
    input  logic [511:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_rdata,
    input  logic [  3:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_rid,
    input  logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_rlast,
    output logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_rready,
    input  logic [  1:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_rresp,
    input  logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid,
    output logic [511:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_wdata,
    output logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_wlast,
    input  logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_wready,
    output logic [ 63:0]                prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb,
    output logic                        prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid,

    // Weight Channels: FTE -> Prefetcher Weight Bank
    input  logic                [top_pkg::PRECISION_COUNT-1:0] weight_channel_req_valid,
    output logic                [top_pkg::PRECISION_COUNT-1:0] weight_channel_req_ready,
    input  WEIGHT_CHANNEL_REQ_t [top_pkg::PRECISION_COUNT-1:0] weight_channel_req,

    output logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp_valid,
    input  logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp_ready,
    output WEIGHT_CHANNEL_RESP_t [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp
);

  // ==================================================================================================================================================
  // Declarations
  // ==================================================================================================================================================

  // Register Bank
  // --------------------------------------------------------------------------------------------

  logic layer_config_in_features_strobe;
  logic [9:0] layer_config_in_features_count;

  logic layer_config_out_features_strobe;
  logic [9:0] layer_config_out_features_count;

  logic layer_config_in_messages_address_lsb_strobe;
  logic [31:0] layer_config_in_messages_address_lsb_value;

  logic layer_config_in_messages_address_msb_strobe;
  logic [1:0] layer_config_in_messages_address_msb_value;

  logic layer_config_adjacency_list_address_lsb_strobe;
  logic [31:0] layer_config_adjacency_list_address_lsb_value;

  logic layer_config_adjacency_list_address_msb_strobe;
  logic [1:0] layer_config_adjacency_list_address_msb_value;

  logic [3:0] layer_config_weights_address_lsb_strobe;
  logic [3:0][31:0] layer_config_weights_address_lsb_value;

  logic [3:0] layer_config_weights_address_msb_strobe;
  logic [3:0][1:0] layer_config_weights_address_msb_value;

  logic layer_config_scale_factors_address_lsb_strobe;
  logic [31:0] layer_config_scale_factors_address_lsb_value;

  logic layer_config_scale_factors_address_msb_strobe;
  logic [1:0] layer_config_scale_factors_address_msb_value;


  // Weight Bank
  // --------------------------------------------------------------------------------------------

  logic [top_pkg::PRECISION_COUNT-1:0] nsb_prefetcher_weight_bank_req_valid;
  logic [top_pkg::PRECISION_COUNT-1:0] nsb_prefetcher_weight_bank_req_ready;

  logic [top_pkg::PRECISION_COUNT-1:0] nsb_prefetcher_weight_bank_resp_valid;
  NSB_PREF_RESP_t [top_pkg::PRECISION_COUNT-1:0] nsb_prefetcher_weight_bank_resp;

  logic [top_pkg::PRECISION_COUNT-1:0] weight_bank_axi_rm_fetch_req_valid;
  logic weight_read_master_fetch_req_ready;
  logic [top_pkg::PRECISION_COUNT-1:0] weight_bank_axi_rm_fetch_req_ready;
  logic           [top_pkg::PRECISION_COUNT-1:0] [AXI_ADDRESS_WIDTH-1:0]                      weight_bank_axi_rm_fetch_start_address;
  logic [top_pkg::PRECISION_COUNT-1:0][$clog2(
MAX_FETCH_REQ_BYTE_COUNT
)-1:0] weight_bank_axi_rm_fetch_byte_count;

  logic weight_read_master_fetch_resp_valid;
  logic [top_pkg::PRECISION_COUNT-1:0] weight_bank_axi_rm_fetch_resp_valid;
  logic [top_pkg::PRECISION_COUNT-1:0] weight_bank_axi_rm_fetch_resp_ready;
  logic weight_read_master_resp_last;
  logic [AXI_DATA_WIDTH-1:0] weight_read_master_resp_data;
  logic [3:0] weight_read_master_resp_axi_id;

  top_pkg::NODE_PRECISION_e active_weight_fetch_precision;

  // ==================================================================================================================================================
  // Instances
  // ==================================================================================================================================================

  // prefetcher_regbank_wrapper prefetcher_regbank_i (
  //     // Clock and Reset
  //     .axi_aclk                       (regbank_clk),
  //     .axi_aresetn                    (regbank_resetn),
  //     .fast_clk                       (core_clk),
  //     .fast_resetn                    (resetn),

  //     // AXI Write Address Channel
  //     .s_axi_awaddr                   (s_axi_awaddr),
  //     .s_axi_awprot                   (s_axi_awprot),
  //     .s_axi_awvalid                  (s_axi_awvalid),
  //     .s_axi_awready                  (s_axi_awready),
  //     .s_axi_wdata                    (s_axi_wdata),
  //     .s_axi_wstrb                    (s_axi_wstrb),
  //     .s_axi_wvalid                   (s_axi_wvalid),
  //     .s_axi_wready                   (s_axi_wready),
  //     .s_axi_araddr                   (s_axi_araddr),
  //     .s_axi_arprot                   (s_axi_arprot),
  //     .s_axi_arvalid                  (s_axi_arvalid),
  //     .s_axi_arready                  (s_axi_arready),
  //     .s_axi_rdata                    (s_axi_rdata),
  //     .s_axi_rresp                    (s_axi_rresp),
  //     .s_axi_rvalid                   (s_axi_rvalid),
  //     .s_axi_rready                   (s_axi_rready),
  //     .s_axi_bresp                    (s_axi_bresp),
  //     .s_axi_bvalid                   (s_axi_bvalid),
  //     .s_axi_bready                   (s_axi_bready),

  //     // User Ports
  //     .layer_config_in_features_count,
  //     .layer_config_out_features_count,
  //     .layer_config_adjacency_list_address_lsb_value,
  //     .layer_config_adjacency_list_address_msb_value,
  //     .layer_config_weights_address_lsb_value,
  //     .layer_config_weights_address_msb_value,
  //     .layer_config_in_messages_address_lsb_value,
  //     .layer_config_in_messages_address_msb_value,
  //     .layer_config_scale_factors_address_lsb_value,
  //     .layer_config_scale_factors_address_msb_value
  // );

  // Weight Bank
  // --------------------------------------------------------------------------------------------

  prefetcher_weight_bank #(
      .PRECISION        (top_pkg::FIXED_8),
      .AXI_ADDRESS_WIDTH(AXI_ADDRESS_WIDTH),
      .AXI_DATA_WIDTH   (AXI_DATA_WIDTH),
      .MAX_FEATURE_COUNT(top_pkg::MAX_FEATURE_COUNT),
      .MAX_FIFO_ROWS(MAX_FIFO_ROWS)
  ) weight_bank_fixed_i (
      .core_clk,
      .resetn,

      .nsb_prefetcher_weight_bank_req_valid(nsb_prefetcher_weight_bank_req_valid[top_pkg::FIXED_8]),
      .nsb_prefetcher_weight_bank_req_ready(nsb_prefetcher_weight_bank_req_ready[top_pkg::FIXED_8]),
      .nsb_prefetcher_weight_bank_req(nsb_prefetcher_req),

      .nsb_prefetcher_weight_bank_resp_valid  (nsb_prefetcher_weight_bank_resp_valid [top_pkg::FIXED_8]),
      .nsb_prefetcher_weight_bank_resp(nsb_prefetcher_weight_bank_resp[top_pkg::FIXED_8]),

      .weight_bank_axi_rm_fetch_req_valid(weight_bank_axi_rm_fetch_req_valid[top_pkg::FIXED_8]),
      .weight_bank_axi_rm_fetch_req_ready(weight_bank_axi_rm_fetch_req_ready[top_pkg::FIXED_8]),
      .weight_bank_axi_rm_fetch_start_address (weight_bank_axi_rm_fetch_start_address [top_pkg::FIXED_8]),
      .weight_bank_axi_rm_fetch_byte_count(weight_bank_axi_rm_fetch_byte_count[top_pkg::FIXED_8]),

      .weight_bank_axi_rm_fetch_resp_valid (weight_bank_axi_rm_fetch_resp_valid[top_pkg::FIXED_8]),
      .weight_bank_axi_rm_fetch_resp_ready (weight_bank_axi_rm_fetch_resp_ready[top_pkg::FIXED_8]),
      .weight_bank_axi_rm_fetch_resp_last  (weight_read_master_resp_last),
      .weight_bank_axi_rm_fetch_resp_data  (weight_read_master_resp_data),
      .weight_bank_axi_rm_fetch_resp_axi_id(weight_read_master_resp_axi_id),

      .weight_channel_req_valid(weight_channel_req_valid[top_pkg::FIXED_8]),
      .weight_channel_req_ready(weight_channel_req_ready[top_pkg::FIXED_8]),
      .weight_channel_req      (weight_channel_req[top_pkg::FIXED_8]),

      .weight_channel_resp_valid(weight_channel_resp_valid[top_pkg::FIXED_8]),
      .weight_channel_resp_ready(weight_channel_resp_ready[top_pkg::FIXED_8]),
      .weight_channel_resp      (weight_channel_resp[top_pkg::FIXED_8]),

      .layer_config_weights_address_lsb_value (layer_config_weights_address_lsb_value [top_pkg::FIXED_8])
  );


  // Weight Bank AXI Read Master
  // --------------------------------------------------------------------------------------------

  axi_read_master #(
      .AXI_ADDRESS_WIDTH(AXI_ADDRESS_WIDTH),
      .AXI_DATA_WIDTH(AXI_DATA_WIDTH),
      .MAX_BYTE_COUNT(MAX_FETCH_REQ_BYTE_COUNT)
  ) weight_read_master_i (
      .core_clk,
      .resetn,

      // Request interface
      .fetch_req_valid    (weight_bank_axi_rm_fetch_req_valid[active_weight_fetch_precision]),
      .fetch_req_ready    (weight_read_master_fetch_req_ready),
      .fetch_start_address(weight_bank_axi_rm_fetch_start_address[active_weight_fetch_precision]),
      .fetch_byte_count   (weight_bank_axi_rm_fetch_byte_count[active_weight_fetch_precision]),

      .fetch_resp_valid (weight_read_master_fetch_resp_valid),
      .fetch_resp_ready (weight_bank_axi_rm_fetch_resp_ready[active_weight_fetch_precision]),
      .fetch_resp_data  (weight_read_master_resp_data),
      .fetch_resp_last  (weight_read_master_resp_last),
      .fetch_resp_axi_id(weight_read_master_resp_axi_id),

      // AXI Read-Only Interface
      .axi_araddr (prefetcher_weight_bank_rm_axi_interconnect_axi_araddr),
      .axi_arburst(prefetcher_weight_bank_rm_axi_interconnect_axi_arburst),
      .axi_arcache(prefetcher_weight_bank_rm_axi_interconnect_axi_arcache),
      .axi_arid   (prefetcher_weight_bank_rm_axi_interconnect_axi_arid),
      .axi_arlen  (prefetcher_weight_bank_rm_axi_interconnect_axi_arlen),
      .axi_arlock (prefetcher_weight_bank_rm_axi_interconnect_axi_arlock),
      .axi_arprot (prefetcher_weight_bank_rm_axi_interconnect_axi_arprot),
      .axi_arqos  (prefetcher_weight_bank_rm_axi_interconnect_axi_arqos),
      .axi_arsize (prefetcher_weight_bank_rm_axi_interconnect_axi_arsize),
      .axi_arvalid(prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid),
      .axi_arready(prefetcher_weight_bank_rm_axi_interconnect_axi_arready),
      .axi_rdata  (prefetcher_weight_bank_rm_axi_interconnect_axi_rdata),
      .axi_rid    (prefetcher_weight_bank_rm_axi_interconnect_axi_rid),
      .axi_rlast  (prefetcher_weight_bank_rm_axi_interconnect_axi_rlast),
      .axi_rvalid (prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid),
      .axi_rready (prefetcher_weight_bank_rm_axi_interconnect_axi_rready),
      .axi_rresp  (prefetcher_weight_bank_rm_axi_interconnect_axi_rresp)
  );

  // ==================================================================================================================================================
  // Logic
  // ==================================================================================================================================================

  // NSB request/responses
  // --------------------------------------------------------------------------------------------

  // TO DO: when edge bank implemented, buffer responses from weight/feature/edge banks and send them to NSB in order

  always_comb begin
    // nsb_prefetcher_feature_bank_req_valid = nsb_prefetcher_req_valid && ((nsb_prefetcher_req.req_opcode == ADJACENCY_LIST) || (nsb_prefetcher_req.req_opcode == MESSAGES) || (nsb_prefetcher_req.req_opcode == SCALE_FACTOR));

    // nsb_prefetcher_req_ready              = (nsb_prefetcher_req.req_opcode == top_pkg::WEIGHTS) ? |(nsb_prefetcher_weight_bank_req_valid & nsb_prefetcher_weight_bank_req_ready)
    //                                         : nsb_prefetcher_feature_bank_req_ready;
    nsb_prefetcher_req_ready              = |(nsb_prefetcher_weight_bank_req_valid[active_weight_fetch_precision] & nsb_prefetcher_weight_bank_req_ready[active_weight_fetch_precision]);

    // Weight bank response should never happen at the same time as feature bank, so simple MUX is enough
    // nsb_prefetcher_resp_valid             = nsb_prefetcher_feature_bank_resp_valid || |nsb_prefetcher_weight_bank_resp_valid;
    nsb_prefetcher_resp_valid = |nsb_prefetcher_weight_bank_resp_valid[active_weight_fetch_precision];

    // nsb_prefetcher_resp                   = |nsb_prefetcher_weight_bank_resp_valid ? nsb_prefetcher_weight_bank_resp [active_weight_fetch_precision]
    //                                         : nsb_prefetcher_feature_bank_resp;
    nsb_prefetcher_resp = nsb_prefetcher_weight_bank_resp[active_weight_fetch_precision];
  end

  // Multi-precision support
  // --------------------------------------------------------------------------------------------

  always_ff @(posedge core_clk or negedge resetn) begin
    if (!resetn) begin
      // active_weight_fetch_precision <= top_pkg::FLOAT_32;
      // active_weight_fetch_precision <= top_pkg::FIXED_16;
      active_weight_fetch_precision <= top_pkg::FIXED_8;

      // Accepting weight fetch request from NSB
    end else begin
      if (nsb_prefetcher_req_valid && nsb_prefetcher_req_ready && nsb_prefetcher_req.req_opcode == WEIGHTS) begin
        active_weight_fetch_precision <= nsb_prefetcher_req.nodeslot_precision;
      end
    end
  end

  // Drive weight bank request
  for (
      genvar precision = top_pkg::FLOAT_32; precision < top_pkg::PRECISION_COUNT; precision++
  ) begin
    assign nsb_prefetcher_weight_bank_req_valid [precision]  = nsb_prefetcher_req_valid && (nsb_prefetcher_req.req_opcode == WEIGHTS) && (precision == nsb_prefetcher_req.nodeslot_precision);

    assign weight_bank_axi_rm_fetch_req_ready [precision] = weight_read_master_fetch_req_ready && (precision == active_weight_fetch_precision);

    assign weight_bank_axi_rm_fetch_resp_valid [precision] = weight_read_master_fetch_resp_valid && (precision == active_weight_fetch_precision);
  end

  // Read-only interfaces
  // --------------------------------------------------------------------------------------------

  always_comb begin
    // Weight Bank
    prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr  = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awburst = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awcache = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awid    = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awlen   = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awlock  = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awprot  = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awqos   = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awsize  = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_bready  = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_wdata   = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_wlast   = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb   = '0;
    prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid  = '0;
  end

  // ==================================================================================================================================================
  // Assertions
  // ==================================================================================================================================================

endmodule
