`timescale 1ps/1ps

import top_pkg::*;
// Top module for the testbench

module top_wrapper_tb #(
    parameter AXI_DATA_WIDTH = 512,
    parameter AXI_ADDR_WIDTH = 30,
    parameter STRB_WIDTH = (AXI_DATA_WIDTH / 8),
    parameter ID_WIDTH = 8
)
(
    input clk,
    input rst,

    input   logic                                                weight_prefetcher_req_valid,
    output  logic                                                weight_prefetcher_req_ready,
    input   NSB_PREF_REQ_t                                       weight_prefetcher_req,
    output  logic                                                weight_prefetcher_resp_valid,
    output  NSB_PREF_RESP_t                                      weight_prefetcher_resp,

    // message -> data_preferrer
    input   logic                                                feature_prefetcher_req_valid,
    output  logic                                                feature_prefetcher_req_ready,
    input   NSB_PREF_REQ_t                                       feature_prefetcher_req,
    output  logic                                                feature_prefetcher_resp_valid,
    output  NSB_PREF_RESP_t                                      feature_prefetcher_resp,

    // Controller -> Transformation Engine Interface
    input   logic                                                nsb_fte_req_valid,
    output  logic                                                nsb_fte_req_ready,
    input   NSB_FTE_REQ_t                                        nsb_fte_req,
    output  logic                                                nsb_fte_resp_valid, // valid only for now
    output  NSB_FTE_RESP_t                                       nsb_fte_resp,

    // Layer Config
    input logic [31:0]  layer_config_out_channel_count,
    input logic [31:0]  layer_config_out_features_count,   
    input logic [1:0]   layer_config_out_features_address_msb_value,
    input logic [AXI_ADDR_WIDTH-2:0] layer_config_out_features_address_lsb_value,
    input logic [AXI_ADDR_WIDTH-1:0] writeback_offset,
    input logic [1:0]  layer_config_activation_function_value,
    input logic [top_pkg::CORE_COUNT*top_pkg::SYSTOLIC_MODULE_COUNT*top_pkg::TRANSFORMATION_ROWS-1:0] [31:0] layer_config_bias_value    
);
// ====================================================================================
// Declarations
// ====================================================================================

// AXI Memory Interconnect -> Memory (Routed to DRAM Controller if `DRAM_CONTROLLER defined)
logic  [7:0]                   c0_ddr4_s_axi_awid;
logic  [33:0]                  c0_ddr4_s_axi_awaddr;
logic  [7:0]                   c0_ddr4_s_axi_awlen;
logic  [2:0]                   c0_ddr4_s_axi_awsize;
logic  [1:0]                   c0_ddr4_s_axi_awburst;
logic  [0:0]                   c0_ddr4_s_axi_awlock;
logic  [3:0]                   c0_ddr4_s_axi_awcache;
logic  [2:0]                   c0_ddr4_s_axi_awprot;
logic  [3:0]                   c0_ddr4_s_axi_awqos;
logic                          c0_ddr4_s_axi_awvalid;
logic                          c0_ddr4_s_axi_awready;
logic  [511:0]                 c0_ddr4_s_axi_wdata;
logic  [63:0]                  c0_ddr4_s_axi_wstrb;
logic                          c0_ddr4_s_axi_wlast;
logic                          c0_ddr4_s_axi_wvalid;
logic                          c0_ddr4_s_axi_wready;
logic [7:0]                    c0_ddr4_s_axi_bid;
logic [1:0]                    c0_ddr4_s_axi_bresp;
logic                          c0_ddr4_s_axi_bvalid;
logic                          c0_ddr4_s_axi_bready;
logic  [7:0]                   c0_ddr4_s_axi_arid;
logic  [33:0]                  c0_ddr4_s_axi_araddr;
logic  [7:0]                   c0_ddr4_s_axi_arlen;
logic  [2:0]                   c0_ddr4_s_axi_arsize;
logic  [1:0]                   c0_ddr4_s_axi_arburst;
logic  [0:0]                   c0_ddr4_s_axi_arlock;
logic  [3:0]                   c0_ddr4_s_axi_arcache;
logic  [2:0]                   c0_ddr4_s_axi_arprot;
logic  [3:0]                   c0_ddr4_s_axi_arqos;
logic                          c0_ddr4_s_axi_arvalid;
logic                          c0_ddr4_s_axi_arready;
logic [7:0]                    c0_ddr4_s_axi_rid;
logic [511:0]                  c0_ddr4_s_axi_rdata;
logic [1:0]                    c0_ddr4_s_axi_rresp;
logic                          c0_ddr4_s_axi_rlast;
logic                          c0_ddr4_s_axi_rvalid;
logic                          c0_ddr4_s_axi_rready;

// ====================================================================================
// Instances
// ====================================================================================

top top_i
(
    .clk                                   (clk),
    .rst                                   (rst),

    .weight_prefetcher_req_valid                (weight_prefetcher_req_valid),
    .weight_prefetcher_req_ready                (weight_prefetcher_req_ready),
    .weight_prefetcher_req                      (weight_prefetcher_req),
    .weight_prefetcher_resp_valid               (weight_prefetcher_resp_valid),
    .weight_prefetcher_resp                     (weight_prefetcher_resp),

    .feature_prefetcher_req_valid               (feature_prefetcher_req_valid),
    .feature_prefetcher_req_ready               (feature_prefetcher_req_ready),
    .feature_prefetcher_req                     (feature_prefetcher_req),
    .feature_prefetcher_resp_valid              (feature_prefetcher_resp_valid),
    .feature_prefetcher_resp                    (feature_prefetcher_resp),

    .nsb_fte_req_valid                          (nsb_fte_req_valid),
    .nsb_fte_req_ready                          (nsb_fte_req_ready),
    .nsb_fte_req                                (nsb_fte_req),
    .nsb_fte_resp_valid                         (nsb_fte_resp_valid),
    .nsb_fte_resp                               (nsb_fte_resp),

    .layer_config_out_channel_count                 (layer_config_out_channel_count),
    .layer_config_out_features_count                (layer_config_out_features_count),
    .layer_config_out_features_address_msb_value    (layer_config_out_features_address_msb_value),
    .layer_config_out_features_address_lsb_value    (layer_config_out_features_address_lsb_value),
    .writeback_offset                               (writeback_offset),
    .layer_config_activation_function_value         (layer_config_activation_function_value),
    .layer_config_bias_value                        (layer_config_bias_value),
        
    .c0_ddr4_s_axi_awid                        (c0_ddr4_s_axi_awid),
    .c0_ddr4_s_axi_awaddr                      (c0_ddr4_s_axi_awaddr),
    .c0_ddr4_s_axi_awlen                       (c0_ddr4_s_axi_awlen),
    .c0_ddr4_s_axi_awsize                      (c0_ddr4_s_axi_awsize),
    .c0_ddr4_s_axi_awburst                     (c0_ddr4_s_axi_awburst),
    .c0_ddr4_s_axi_awlock                      (c0_ddr4_s_axi_awlock),
    .c0_ddr4_s_axi_awcache                     (c0_ddr4_s_axi_awcache),
    .c0_ddr4_s_axi_awprot                      (c0_ddr4_s_axi_awprot),
    .c0_ddr4_s_axi_awqos                       (c0_ddr4_s_axi_awqos),
    .c0_ddr4_s_axi_awvalid                     (c0_ddr4_s_axi_awvalid),
    .c0_ddr4_s_axi_awready                     (c0_ddr4_s_axi_awready),
    .c0_ddr4_s_axi_wdata                       (c0_ddr4_s_axi_wdata),
    .c0_ddr4_s_axi_wstrb                       (c0_ddr4_s_axi_wstrb),
    .c0_ddr4_s_axi_wlast                       (c0_ddr4_s_axi_wlast),
    .c0_ddr4_s_axi_wvalid                      (c0_ddr4_s_axi_wvalid),
    .c0_ddr4_s_axi_wready                      (c0_ddr4_s_axi_wready),
    .c0_ddr4_s_axi_bid                         (c0_ddr4_s_axi_bid),
    .c0_ddr4_s_axi_bresp                       (c0_ddr4_s_axi_bresp),
    .c0_ddr4_s_axi_bvalid                      (c0_ddr4_s_axi_bvalid),
    .c0_ddr4_s_axi_bready                      (c0_ddr4_s_axi_bready),
    .c0_ddr4_s_axi_arid                        (c0_ddr4_s_axi_arid),
    .c0_ddr4_s_axi_araddr                      (c0_ddr4_s_axi_araddr),
    .c0_ddr4_s_axi_arlen                       (c0_ddr4_s_axi_arlen),
    .c0_ddr4_s_axi_arsize                      (c0_ddr4_s_axi_arsize),
    .c0_ddr4_s_axi_arburst                     (c0_ddr4_s_axi_arburst),
    .c0_ddr4_s_axi_arlock                      (c0_ddr4_s_axi_arlock),
    .c0_ddr4_s_axi_arcache                     (c0_ddr4_s_axi_arcache),
    .c0_ddr4_s_axi_arprot                      (c0_ddr4_s_axi_arprot),
    .c0_ddr4_s_axi_arqos                       (c0_ddr4_s_axi_arqos),
    .c0_ddr4_s_axi_arvalid                     (c0_ddr4_s_axi_arvalid),
    .c0_ddr4_s_axi_arready                     (c0_ddr4_s_axi_arready),
    .c0_ddr4_s_axi_rid                         (c0_ddr4_s_axi_rid),
    .c0_ddr4_s_axi_rdata                       (c0_ddr4_s_axi_rdata),
    .c0_ddr4_s_axi_rresp                       (c0_ddr4_s_axi_rresp),
    .c0_ddr4_s_axi_rlast                       (c0_ddr4_s_axi_rlast),
    .c0_ddr4_s_axi_rvalid                      (c0_ddr4_s_axi_rvalid),
    .c0_ddr4_s_axi_rready                      (c0_ddr4_s_axi_rready)
);


axi_ram #(
    .DATA_WIDTH(AXI_DATA_WIDTH),
    .ADDR_WIDTH(AXI_ADDR_WIDTH),
    .ID_WIDTH(8)
) ram_model (
    .clk                    (clk),
    .rst                    (rst),

    .s_axi_awid             (c0_ddr4_s_axi_awid),
    .s_axi_awaddr           (c0_ddr4_s_axi_awaddr),
    .s_axi_awlen            (c0_ddr4_s_axi_awlen),
    .s_axi_awsize           (c0_ddr4_s_axi_awsize),
    .s_axi_awburst          (c0_ddr4_s_axi_awburst),
    .s_axi_awlock           (c0_ddr4_s_axi_awlock),
    .s_axi_awcache          (c0_ddr4_s_axi_awcache),
    .s_axi_awprot           (c0_ddr4_s_axi_awprot),
    .s_axi_awvalid          (c0_ddr4_s_axi_awvalid),
    .s_axi_awready          (c0_ddr4_s_axi_awready),
    .s_axi_wdata            (c0_ddr4_s_axi_wdata),
    .s_axi_wstrb            (c0_ddr4_s_axi_wstrb),
    .s_axi_wlast            (c0_ddr4_s_axi_wlast),
    .s_axi_wvalid           (c0_ddr4_s_axi_wvalid),
    .s_axi_wready           (c0_ddr4_s_axi_wready),
    .s_axi_bid              (c0_ddr4_s_axi_bid),
    .s_axi_bresp            (c0_ddr4_s_axi_bresp),
    .s_axi_bvalid           (c0_ddr4_s_axi_bvalid),
    .s_axi_bready           (c0_ddr4_s_axi_bready),
    .s_axi_arid             (c0_ddr4_s_axi_arid),
    .s_axi_araddr           (c0_ddr4_s_axi_araddr),
    .s_axi_arlen            (c0_ddr4_s_axi_arlen),
    .s_axi_arsize           (c0_ddr4_s_axi_arsize),
    .s_axi_arburst          (c0_ddr4_s_axi_arburst),
    .s_axi_arlock           (c0_ddr4_s_axi_arlock),
    .s_axi_arcache          (c0_ddr4_s_axi_arcache),
    .s_axi_arprot           (c0_ddr4_s_axi_arprot),
    .s_axi_arvalid          (c0_ddr4_s_axi_arvalid),
    .s_axi_arready          (c0_ddr4_s_axi_arready),
    .s_axi_rid              (c0_ddr4_s_axi_rid),
    .s_axi_rdata            (c0_ddr4_s_axi_rdata),
    .s_axi_rresp            (c0_ddr4_s_axi_rresp),
    .s_axi_rlast            (c0_ddr4_s_axi_rlast),
    .s_axi_rvalid           (c0_ddr4_s_axi_rvalid),
    .s_axi_rready           (c0_ddr4_s_axi_rready)
);

endmodule
