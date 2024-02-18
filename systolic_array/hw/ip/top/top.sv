`timescale 1ps/1ps

import top_pkg::*;

module top #(
    parameter DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 34,
    parameter STRB_WIDTH = (DATA_WIDTH / 8),
    parameter ID_WIDTH = 4
)
(
    input clk,
    input rst,

    // input logic                           sys_clk,
    // input logic                           sys_rst, //Common port for all controllers

    // input  logic                          regbank_clk,
    // input  logic                          regbank_resetn,

    // AXI Memory Interconnect -> Memory (Routed to DRAM Controller if `DRAM_CONTROLLER defined)
    // output logic  [7:0]                   c0_ddr4_s_axi_awid,
    // output logic  [33:0]                  c0_ddr4_s_axi_awaddr,
    // output logic  [7:0]                   c0_ddr4_s_axi_awlen,
    // output logic  [2:0]                   c0_ddr4_s_axi_awsize,
    // output logic  [1:0]                   c0_ddr4_s_axi_awburst,
    // output logic  [0:0]                   c0_ddr4_s_axi_awlock,
    // output logic  [3:0]                   c0_ddr4_s_axi_awcache,
    // output logic  [2:0]                   c0_ddr4_s_axi_awprot,
    // output logic  [3:0]                   c0_ddr4_s_axi_awqos,
    // output logic                          c0_ddr4_s_axi_awvalid,
    // input  logic                          c0_ddr4_s_axi_awready,
    // output logic  [511:0]                 c0_ddr4_s_axi_wdata,
    // output logic  [63:0]                  c0_ddr4_s_axi_wstrb,
    // output logic                          c0_ddr4_s_axi_wlast,
    // output logic                          c0_ddr4_s_axi_wvalid,
    // input  logic                          c0_ddr4_s_axi_wready,
    // input  logic [7:0]                    c0_ddr4_s_axi_bid,
    // input  logic [1:0]                    c0_ddr4_s_axi_bresp,
    // input  logic                          c0_ddr4_s_axi_bvalid,
    // output logic                          c0_ddr4_s_axi_bready,
    // output logic  [7:0]                   c0_ddr4_s_axi_arid,
    // output logic  [33:0]                  c0_ddr4_s_axi_araddr,
    // output logic  [7:0]                   c0_ddr4_s_axi_arlen,
    // output logic  [2:0]                   c0_ddr4_s_axi_arsize,
    // output logic  [1:0]                   c0_ddr4_s_axi_arburst,
    // output logic  [0:0]                   c0_ddr4_s_axi_arlock,
    // output logic  [3:0]                   c0_ddr4_s_axi_arcache,
    // output logic  [2:0]                   c0_ddr4_s_axi_arprot,
    // output logic  [3:0]                   c0_ddr4_s_axi_arqos,
    // output logic                          c0_ddr4_s_axi_arvalid,
    // input  logic                          c0_ddr4_s_axi_arready,
    // input  logic [7:0]                    c0_ddr4_s_axi_rid,
    // input  logic [511:0]                  c0_ddr4_s_axi_rdata,
    // input  logic [1:0]                    c0_ddr4_s_axi_rresp,
    // input  logic                          c0_ddr4_s_axi_rlast,
    // input  logic                          c0_ddr4_s_axi_rvalid,
    // output logic                          c0_ddr4_s_axi_rready,

    // message -> weight_preferrer
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
    input logic [9:0]  layer_config_in_features_count,
    input logic [9:0]  layer_config_out_features_count,                                   
    input logic [1:0]  layer_config_activation_function_value,
    input logic [31:0] layer_config_bias_value,
    input logic [31:0] layer_config_leaky_relu_alpha_value,
    input logic [1:0]  layer_config_out_features_address_msb_value,
    input logic [31:0] layer_config_out_features_address_lsb_value,
    input logic [0:0]  ctrl_buffering_enable_value,
    input logic [0:0]  ctrl_writeback_enable_value
);
// ====================================================================================
// Declarations
// ====================================================================================
// Prefetcher Weight Bank Read Master -> AXI Memory Interconnect (Read Only)
logic [33:0]                       prefetcher_weight_bank_rm_axi_interconnect_axi_araddr;
logic [1:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arburst;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arcache;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arid;
logic [7:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arlen;
logic [0:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arlock;
logic [2:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arprot;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arqos;
logic [2:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_arsize;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_arready;
logic [33:0]                       prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr;
logic [1:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awburst;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awcache;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awid;
logic [7:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awlen;
logic [0:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awlock;
logic [2:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awprot;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awqos;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_awready;
logic [2:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_awsize;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_bid;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_bready;
logic [1:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_bresp;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid;
logic [511:0]                      prefetcher_weight_bank_rm_axi_interconnect_axi_rdata;
logic [3:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_rid;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_rlast;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_rready;
logic [1:0]                        prefetcher_weight_bank_rm_axi_interconnect_axi_rresp;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid;
logic [511:0]                      prefetcher_weight_bank_rm_axi_interconnect_axi_wdata;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_wlast;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_wready;
logic [63:0]                       prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb;
logic                              prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid;

// Prefetcher feature Bank Read Master -> AXI Memory Interconnect (Read Only)
logic [33:0]                       prefetcher_feature_bank_rm_axi_interconnect_axi_araddr;
logic [1:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arburst;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arcache;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arid;
logic [7:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arlen;
logic [0:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arlock;
logic [2:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arprot;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arqos;
logic [2:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_arsize;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_arvalid;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_arready;
logic [33:0]                       prefetcher_feature_bank_rm_axi_interconnect_axi_awaddr;
logic [1:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awburst;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awcache;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awid;
logic [7:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awlen;
logic [0:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awlock;
logic [2:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awprot;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awqos;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_awready;
logic [2:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_awsize;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_awvalid;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_bid;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_bready;
logic [1:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_bresp;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_bvalid;
logic [511:0]                      prefetcher_feature_bank_rm_axi_interconnect_axi_rdata;
logic [3:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_rid;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_rlast;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_rready;
logic [1:0]                        prefetcher_feature_bank_rm_axi_interconnect_axi_rresp;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_rvalid;
logic [511:0]                      prefetcher_feature_bank_rm_axi_interconnect_axi_wdata;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_wlast;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_wready;
logic [63:0]                       prefetcher_feature_bank_rm_axi_interconnect_axi_wstrb;
logic                              prefetcher_feature_bank_rm_axi_interconnect_axi_wvalid;

// Feature Transformation Engine -> AXI Memory Interconnect (Write Only)
logic [33:0]                       transformation_engine_axi_interconnect_axi_araddr;
logic [1:0]                        transformation_engine_axi_interconnect_axi_arburst;
logic [3:0]                        transformation_engine_axi_interconnect_axi_arcache;
logic [3:0]                        transformation_engine_axi_interconnect_axi_arid;
logic [7:0]                        transformation_engine_axi_interconnect_axi_arlen;
logic [0:0]                        transformation_engine_axi_interconnect_axi_arlock;
logic [2:0]                        transformation_engine_axi_interconnect_axi_arprot;
logic [3:0]                        transformation_engine_axi_interconnect_axi_arqos;
logic [2:0]                        transformation_engine_axi_interconnect_axi_arsize;
logic                              transformation_engine_axi_interconnect_axi_arvalid;
logic                              transformation_engine_axi_interconnect_axi_arready;
logic [33:0]                       transformation_engine_axi_interconnect_axi_awaddr;
logic [1:0]                        transformation_engine_axi_interconnect_axi_awburst;
logic [3:0]                        transformation_engine_axi_interconnect_axi_awcache;
logic [3:0]                        transformation_engine_axi_interconnect_axi_awid;
logic [7:0]                        transformation_engine_axi_interconnect_axi_awlen;
logic [0:0]                        transformation_engine_axi_interconnect_axi_awlock;
logic [2:0]                        transformation_engine_axi_interconnect_axi_awprot;
logic [3:0]                        transformation_engine_axi_interconnect_axi_awqos;
logic                              transformation_engine_axi_interconnect_axi_awready;
logic [2:0]                        transformation_engine_axi_interconnect_axi_awsize;
logic                              transformation_engine_axi_interconnect_axi_awvalid;
logic [3:0]                        transformation_engine_axi_interconnect_axi_bid;
logic                              transformation_engine_axi_interconnect_axi_bready;
logic [1:0]                        transformation_engine_axi_interconnect_axi_bresp;
logic                              transformation_engine_axi_interconnect_axi_bvalid;
logic [511:0]                      transformation_engine_axi_interconnect_axi_rdata;
logic [3:0]                        transformation_engine_axi_interconnect_axi_rid;
logic                              transformation_engine_axi_interconnect_axi_rlast;
logic                              transformation_engine_axi_interconnect_axi_rready;
logic [1:0]                        transformation_engine_axi_interconnect_axi_rresp;
logic                              transformation_engine_axi_interconnect_axi_rvalid;
logic [511:0]                      transformation_engine_axi_interconnect_axi_wdata;
logic                              transformation_engine_axi_interconnect_axi_wlast;
logic                              transformation_engine_axi_interconnect_axi_wready;
logic [63:0]                       transformation_engine_axi_interconnect_axi_wstrb;
logic                              transformation_engine_axi_interconnect_axi_wvalid;

logic S00_AXI_ARESET_OUT_N;
logic S01_AXI_ARESET_OUT_N;
logic S02_AXI_ARESET_OUT_N;
logic M00_AXI_ARESET_OUT_N;

// Weight Channel: FTE -> Prefetcher Weight Bank (REQ)
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_req_valid;
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_req_ready;
WEIGHT_CHANNEL_REQ_t  [top_pkg::PRECISION_COUNT-1:0] weight_channel_req;
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp_valid;
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp_ready;
WEIGHT_CHANNEL_RESP_t [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp;

// Feature Channels: FTE -> Prefetcher Feature Bank (REQ)
logic                 [top_pkg::PRECISION_COUNT-1:0] feature_channel_req_valid;
logic                 [top_pkg::PRECISION_COUNT-1:0] feature_channel_req_ready;
FEATURE_CHANNEL_REQ_t [top_pkg::PRECISION_COUNT-1:0] feature_channel_req;
logic                 [top_pkg::PRECISION_COUNT-1:0] feature_channel_resp_valid;
logic                 [top_pkg::PRECISION_COUNT-1:0] feature_channel_resp_ready;
FEATURE_CHANNEL_REQ_t [top_pkg::PRECISION_COUNT-1:0] feature_channel_resp;

// ====================================================================================
// AXI Memory 
// ====================================================================================
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

axi_interface axi_ram (
    .clk                        (clk),
    .rst                        (rst),

    .axi_awid                   (c0_ddr4_s_axi_awid),
    .axi_awaddr                 (c0_ddr4_s_axi_awaddr),
    .axi_awlen                  (c0_ddr4_s_axi_awlen),
    .axi_awsize                 (c0_ddr4_s_axi_awsize),
    .axi_awburst                (c0_ddr4_s_axi_awburst),
    .axi_awlock                 (c0_ddr4_s_axi_awlock),
    .axi_awcache                (c0_ddr4_s_axi_awcache),
    .axi_awprot                 (c0_ddr4_s_axi_awprot),
    .axi_awqos                  (c0_ddr4_s_axi_awqos), // not used 
    .axi_awregion               (), // not used
    .axi_awvalid                (c0_ddr4_s_axi_awvalid),
    .axi_awready                (c0_ddr4_s_axi_awready),
    .axi_wdata                  (c0_ddr4_s_axi_wdata),
    .axi_wstrb                  (c0_ddr4_s_axi_wstrb),
    .axi_wlast                  (c0_ddr4_s_axi_wlast),
    .axi_wvalid                 (c0_ddr4_s_axi_wvalid),
    .axi_wready                 (c0_ddr4_s_axi_wready),
    .axi_bid                    (c0_ddr4_s_axi_bid),
    .axi_bresp                  (c0_ddr4_s_axi_bresp),
    .axi_bvalid                 (c0_ddr4_s_axi_bvalid),
    .axi_bready                 (c0_ddr4_s_axi_bready),
    .axi_arid                   (c0_ddr4_s_axi_arid),
    .axi_araddr                 (c0_ddr4_s_axi_araddr),
    .axi_arlen                  (c0_ddr4_s_axi_arlen),
    .axi_arsize                 (c0_ddr4_s_axi_arsize),
    .axi_arburst                (c0_ddr4_s_axi_arburst),
    .axi_arlock                 (c0_ddr4_s_axi_arlock),
    .axi_arcache                (c0_ddr4_s_axi_arcache),
    .axi_arprot                 (c0_ddr4_s_axi_arprot),
    .axi_arqos                  (c0_ddr4_s_axi_arqos), // not used prefetcher_weight_bank_rm_axi_interconnect_axi_arqos
    .axi_arregion               (), // not used
    .axi_arvalid                (c0_ddr4_s_axi_arvalid),
    .axi_arready                (c0_ddr4_s_axi_arready),
    .axi_rid                    (c0_ddr4_s_axi_rid),
    .axi_rdata                  (c0_ddr4_s_axi_rdata),
    .axi_rresp                  (c0_ddr4_s_axi_rresp),
    .axi_rlast                  (c0_ddr4_s_axi_rlast),
    .axi_rvalid                 (c0_ddr4_s_axi_rvalid),
    .axi_rready                 (c0_ddr4_s_axi_rready)
);


// axi_interface axi_ram (
//     .clk                        (clk),
//     .rst                        (rst),

//     .axi_awid                   (prefetcher_weight_bank_rm_axi_interconnect_axi_awid),
//     .axi_awaddr                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr),
//     .axi_awlen                  (prefetcher_weight_bank_rm_axi_interconnect_axi_awlen),
//     .axi_awsize                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awsize),
//     .axi_awburst                (prefetcher_weight_bank_rm_axi_interconnect_axi_awburst),
//     .axi_awlock                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awlock),
//     .axi_awcache                (prefetcher_weight_bank_rm_axi_interconnect_axi_awcache),
//     .axi_awprot                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awprot),
//     .axi_awqos                  (prefetcher_weight_bank_rm_axi_interconnect_axi_awqos),
//     .axi_awregion               (), // not used
//     .axi_awvalid                (prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid),
//     .axi_awready                (prefetcher_weight_bank_rm_axi_interconnect_axi_awready),
//     .axi_wdata                  (prefetcher_weight_bank_rm_axi_interconnect_axi_wdata),
//     .axi_wstrb                  (prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb),
//     .axi_wlast                  (prefetcher_weight_bank_rm_axi_interconnect_axi_wlast),
//     .axi_wvalid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid),
//     .axi_wready                 (prefetcher_weight_bank_rm_axi_interconnect_axi_wready),
//     .axi_bid                    (prefetcher_weight_bank_rm_axi_interconnect_axi_bid),
//     .axi_bresp                  (prefetcher_weight_bank_rm_axi_interconnect_axi_bresp),
//     .axi_bvalid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid),
//     .axi_bready                 (prefetcher_weight_bank_rm_axi_interconnect_axi_bready),
//     .axi_arid                   (prefetcher_weight_bank_rm_axi_interconnect_axi_arid),
//     .axi_araddr                 (prefetcher_weight_bank_rm_axi_interconnect_axi_araddr),
//     .axi_arlen                  (prefetcher_weight_bank_rm_axi_interconnect_axi_arlen),
//     .axi_arsize                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arsize),
//     .axi_arburst                (prefetcher_weight_bank_rm_axi_interconnect_axi_arburst),
//     .axi_arlock                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arlock),
//     .axi_arcache                (prefetcher_weight_bank_rm_axi_interconnect_axi_arcache),
//     .axi_arprot                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arprot),
//     .axi_arqos                  (prefetcher_weight_bank_rm_axi_interconnect_axi_arqos),
//     .axi_arregion               (), // not used
//     .axi_arvalid                (prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid),
//     .axi_arready                (prefetcher_weight_bank_rm_axi_interconnect_axi_arready),
//     .axi_rid                    (prefetcher_weight_bank_rm_axi_interconnect_axi_rid),
//     .axi_rdata                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rdata),
//     .axi_rresp                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rresp),
//     .axi_rlast                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rlast),
//     .axi_rvalid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid),
//     .axi_rready                 (prefetcher_weight_bank_rm_axi_interconnect_axi_rready)
// );

// ====================================================================================
// Prefetcher
// ====================================================================================

// prefetcher #(
//     .FETCH_TAG_COUNT (top_pkg::MESSAGE_CHANNEL_COUNT)
// ) prefetcher_i (
prefetcher prefetcher_weight_i (
    .core_clk                                                  (clk),
    .resetn                                                    (!rst),

    // .regbank_clk                                               (regbank_clk),
    // .regbank_resetn                                            (regbank_resetn),

    // Node Scoreboard -> Prefetcher Interface
    .nsb_prefetcher_req_valid                                  (weight_prefetcher_req_valid),
    .nsb_prefetcher_req_ready                                  (weight_prefetcher_req_ready),
    .nsb_prefetcher_req                                        (weight_prefetcher_req),
    .nsb_prefetcher_resp_valid                                 (weight_prefetcher_resp_valid),
    .nsb_prefetcher_resp                                       (weight_prefetcher_resp),

    // Regbank Slave AXI interface
    // .s_axi_awaddr                                              (axil_interconnect_m_axi_awaddr     [127:96]),
    // .s_axi_wdata                                               (axil_interconnect_m_axi_wdata      [127:96]),
    // .s_axi_araddr                                              (axil_interconnect_m_axi_araddr     [127:96]),
    // .s_axi_rdata                                               (axil_interconnect_m_axi_rdata      [127:96]),
    // .s_axi_awprot                                              (axil_interconnect_m_axi_awprot     [11:9]),
    // .s_axi_arprot                                              (axil_interconnect_m_axi_arprot     [11:9]),
    // .s_axi_awvalid                                             (axil_interconnect_m_axi_awvalid    [3:3]),
    // .s_axi_awready                                             (axil_interconnect_m_axi_awready    [3:3]),
    // .s_axi_wvalid                                              (axil_interconnect_m_axi_wvalid     [3:3]),
    // .s_axi_wready                                              (axil_interconnect_m_axi_wready     [3:3]),
    // .s_axi_bvalid                                              (axil_interconnect_m_axi_bvalid     [3:3]),
    // .s_axi_bready                                              (axil_interconnect_m_axi_bready     [3:3]),
    // .s_axi_arvalid                                             (axil_interconnect_m_axi_arvalid    [3:3]),
    // .s_axi_arready                                             (axil_interconnect_m_axi_arready    [3:3]),
    // .s_axi_rvalid                                              (axil_interconnect_m_axi_rvalid     [3:3]),
    // .s_axi_rready                                              (axil_interconnect_m_axi_rready     [3:3]),
    // .s_axi_wstrb                                               (axil_interconnect_m_axi_wstrb      [15:12]),
    // .s_axi_bresp                                               (axil_interconnect_m_axi_bresp      [7:6]),
    // .s_axi_rresp                                               (axil_interconnect_m_axi_rresp      [7:6]),

    // Prefetcher Weights RM -> AXI Memory Interconnect
    .prefetcher_weight_bank_rm_axi_interconnect_axi_araddr     (prefetcher_weight_bank_rm_axi_interconnect_axi_araddr),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arburst    (prefetcher_weight_bank_rm_axi_interconnect_axi_arburst),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arcache    (prefetcher_weight_bank_rm_axi_interconnect_axi_arcache),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arid       (prefetcher_weight_bank_rm_axi_interconnect_axi_arid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arlen      (prefetcher_weight_bank_rm_axi_interconnect_axi_arlen),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arlock     (prefetcher_weight_bank_rm_axi_interconnect_axi_arlock),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arprot     (prefetcher_weight_bank_rm_axi_interconnect_axi_arprot),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arqos      (prefetcher_weight_bank_rm_axi_interconnect_axi_arqos),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arsize     (prefetcher_weight_bank_rm_axi_interconnect_axi_arsize),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid    (prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arready    (prefetcher_weight_bank_rm_axi_interconnect_axi_arready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr     (prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awburst    (prefetcher_weight_bank_rm_axi_interconnect_axi_awburst),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awcache    (prefetcher_weight_bank_rm_axi_interconnect_axi_awcache),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awid       (prefetcher_weight_bank_rm_axi_interconnect_axi_awid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awlen      (prefetcher_weight_bank_rm_axi_interconnect_axi_awlen),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awlock     (prefetcher_weight_bank_rm_axi_interconnect_axi_awlock),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awprot     (prefetcher_weight_bank_rm_axi_interconnect_axi_awprot),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awqos      (prefetcher_weight_bank_rm_axi_interconnect_axi_awqos),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awready    (prefetcher_weight_bank_rm_axi_interconnect_axi_awready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awsize     (prefetcher_weight_bank_rm_axi_interconnect_axi_awsize),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid    (prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bid        (prefetcher_weight_bank_rm_axi_interconnect_axi_bid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bready     (prefetcher_weight_bank_rm_axi_interconnect_axi_bready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bresp      (prefetcher_weight_bank_rm_axi_interconnect_axi_bresp),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid     (prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rdata      (prefetcher_weight_bank_rm_axi_interconnect_axi_rdata),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rid        (prefetcher_weight_bank_rm_axi_interconnect_axi_rid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rlast      (prefetcher_weight_bank_rm_axi_interconnect_axi_rlast),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rready     (prefetcher_weight_bank_rm_axi_interconnect_axi_rready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rresp      (prefetcher_weight_bank_rm_axi_interconnect_axi_rresp),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid     (prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wdata      (prefetcher_weight_bank_rm_axi_interconnect_axi_wdata),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wlast      (prefetcher_weight_bank_rm_axi_interconnect_axi_wlast),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wready     (prefetcher_weight_bank_rm_axi_interconnect_axi_wready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb      (prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid     (prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid),

    .weight_channel_req_valid                                  (weight_channel_req_valid),
    .weight_channel_req_ready                                  (weight_channel_req_ready),
    .weight_channel_req                                        (weight_channel_req),

    .weight_channel_resp_valid                                 (weight_channel_resp_valid),
    .weight_channel_resp_ready                                 (weight_channel_resp_ready),
    .weight_channel_resp                                       (weight_channel_resp)
);

// ====================================================================================
// Systolic Array
// ====================================================================================
prefetcher prefetcher_feature_i (
    .core_clk                                                  (clk),
    .resetn                                                    (!rst),

    // .regbank_clk                                               (regbank_clk),
    // .regbank_resetn                                            (regbank_resetn),

    // Node Scoreboard -> Prefetcher Interface
    .nsb_prefetcher_req_valid                                  (feature_prefetcher_req_valid),
    .nsb_prefetcher_req_ready                                  (feature_prefetcher_req_ready),
    .nsb_prefetcher_req                                        (feature_prefetcher_req),
    .nsb_prefetcher_resp_valid                                 (feature_prefetcher_resp_valid),
    .nsb_prefetcher_resp                                       (feature_prefetcher_resp),

    // Regbank Slave AXI interface
    // .s_axi_awaddr                                              (axil_interconnect_m_axi_awaddr     [127:96]),
    // .s_axi_wdata                                               (axil_interconnect_m_axi_wdata      [127:96]),
    // .s_axi_araddr                                              (axil_interconnect_m_axi_araddr     [127:96]),
    // .s_axi_rdata                                               (axil_interconnect_m_axi_rdata      [127:96]),
    // .s_axi_awprot                                              (axil_interconnect_m_axi_awprot     [11:9]),
    // .s_axi_arprot                                              (axil_interconnect_m_axi_arprot     [11:9]),
    // .s_axi_awvalid                                             (axil_interconnect_m_axi_awvalid    [3:3]),
    // .s_axi_awready                                             (axil_interconnect_m_axi_awready    [3:3]),
    // .s_axi_wvalid                                              (axil_interconnect_m_axi_wvalid     [3:3]),
    // .s_axi_wready                                              (axil_interconnect_m_axi_wready     [3:3]),
    // .s_axi_bvalid                                              (axil_interconnect_m_axi_bvalid     [3:3]),
    // .s_axi_bready                                              (axil_interconnect_m_axi_bready     [3:3]),
    // .s_axi_arvalid                                             (axil_interconnect_m_axi_arvalid    [3:3]),
    // .s_axi_arready                                             (axil_interconnect_m_axi_arready    [3:3]),
    // .s_axi_rvalid                                              (axil_interconnect_m_axi_rvalid     [3:3]),
    // .s_axi_rready                                              (axil_interconnect_m_axi_rready     [3:3]),
    // .s_axi_wstrb                                               (axil_interconnect_m_axi_wstrb      [15:12]),
    // .s_axi_bresp                                               (axil_interconnect_m_axi_bresp      [7:6]),
    // .s_axi_rresp                                               (axil_interconnect_m_axi_rresp      [7:6]),

    // Prefetcher Weights RM -> AXI Memory Interconnect
    .prefetcher_weight_bank_rm_axi_interconnect_axi_araddr     (prefetcher_feature_bank_rm_axi_interconnect_axi_araddr),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arburst    (prefetcher_feature_bank_rm_axi_interconnect_axi_arburst),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arcache    (prefetcher_feature_bank_rm_axi_interconnect_axi_arcache),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arid       (prefetcher_feature_bank_rm_axi_interconnect_axi_arid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arlen      (prefetcher_feature_bank_rm_axi_interconnect_axi_arlen),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arlock     (prefetcher_feature_bank_rm_axi_interconnect_axi_arlock),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arprot     (prefetcher_feature_bank_rm_axi_interconnect_axi_arprot),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arqos      (prefetcher_feature_bank_rm_axi_interconnect_axi_arqos),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arsize     (prefetcher_feature_bank_rm_axi_interconnect_axi_arsize),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid    (prefetcher_feature_bank_rm_axi_interconnect_axi_arvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_arready    (prefetcher_feature_bank_rm_axi_interconnect_axi_arready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr     (prefetcher_feature_bank_rm_axi_interconnect_axi_awaddr),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awburst    (prefetcher_feature_bank_rm_axi_interconnect_axi_awburst),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awcache    (prefetcher_feature_bank_rm_axi_interconnect_axi_awcache),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awid       (prefetcher_feature_bank_rm_axi_interconnect_axi_awid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awlen      (prefetcher_feature_bank_rm_axi_interconnect_axi_awlen),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awlock     (prefetcher_feature_bank_rm_axi_interconnect_axi_awlock),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awprot     (prefetcher_feature_bank_rm_axi_interconnect_axi_awprot),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awqos      (prefetcher_feature_bank_rm_axi_interconnect_axi_awqos),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awready    (prefetcher_feature_bank_rm_axi_interconnect_axi_awready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awsize     (prefetcher_feature_bank_rm_axi_interconnect_axi_awsize),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid    (prefetcher_feature_bank_rm_axi_interconnect_axi_awvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bid        (prefetcher_feature_bank_rm_axi_interconnect_axi_bid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bready     (prefetcher_feature_bank_rm_axi_interconnect_axi_bready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bresp      (prefetcher_feature_bank_rm_axi_interconnect_axi_bresp),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid     (prefetcher_feature_bank_rm_axi_interconnect_axi_bvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rdata      (prefetcher_feature_bank_rm_axi_interconnect_axi_rdata),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rid        (prefetcher_feature_bank_rm_axi_interconnect_axi_rid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rlast      (prefetcher_feature_bank_rm_axi_interconnect_axi_rlast),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rready     (prefetcher_feature_bank_rm_axi_interconnect_axi_rready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rresp      (prefetcher_feature_bank_rm_axi_interconnect_axi_rresp),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid     (prefetcher_feature_bank_rm_axi_interconnect_axi_rvalid),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wdata      (prefetcher_feature_bank_rm_axi_interconnect_axi_wdata),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wlast      (prefetcher_feature_bank_rm_axi_interconnect_axi_wlast),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wready     (prefetcher_feature_bank_rm_axi_interconnect_axi_wready),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb      (prefetcher_feature_bank_rm_axi_interconnect_axi_wstrb),
    .prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid     (prefetcher_feature_bank_rm_axi_interconnect_axi_wvalid),

    .weight_channel_req_valid                                  (feature_channel_req_valid),
    .weight_channel_req_ready                                  (feature_channel_req_ready),
    .weight_channel_req                                        (feature_channel_req),

    .weight_channel_resp_valid                                 (feature_channel_resp_valid),
    .weight_channel_resp_ready                                 (feature_channel_resp_ready),
    .weight_channel_resp                                       (feature_channel_resp)
);

// ====================================================================================
// Transformation Engine
// ====================================================================================
feature_transformation_engine transformation_engine_i (
    .core_clk                                           (clk),
    .resetn                                             (!rst),

    .regbank_clk                                               (regbank_clk),
    .regbank_resetn                                            (regbank_resetn),

    // // AXI-L interface
    // .s_axi_awaddr                                       (axil_interconnect_m_axi_awaddr     [63:32]), // input
    // .s_axi_awprot                                       (axil_interconnect_m_axi_awprot     [5:3]), // input
    // .s_axi_awvalid                                      (axil_interconnect_m_axi_awvalid    [1:1]), // input
    // .s_axi_awready                                      (axil_interconnect_m_axi_awready    [1:1]), // output
    // .s_axi_wdata                                        (axil_interconnect_m_axi_wdata      [63:32]), // input
    // .s_axi_wstrb                                        (axil_interconnect_m_axi_wstrb      [7:4]), // input
    // .s_axi_wvalid                                       (axil_interconnect_m_axi_wvalid     [1:1]), // input
    // .s_axi_wready                                       (axil_interconnect_m_axi_wready     [1:1]), // output
    // .s_axi_araddr                                       (axil_interconnect_m_axi_araddr     [63:32]), // input
    // .s_axi_arprot                                       (axil_interconnect_m_axi_arprot     [5:3]), // input
    // .s_axi_arvalid                                      (axil_interconnect_m_axi_arvalid    [1:1]), // input
    // .s_axi_arready                                      (axil_interconnect_m_axi_arready    [1:1]), // output
    // .s_axi_rdata                                        (axil_interconnect_m_axi_rdata      [63:32]), // output
    // .s_axi_rresp                                        (axil_interconnect_m_axi_rresp      [3:2]), // output
    // .s_axi_rvalid                                       (axil_interconnect_m_axi_rvalid     [1:1]), // output
    // .s_axi_rready                                       (axil_interconnect_m_axi_rready     [1:1]), // input
    // .s_axi_bresp                                        (axil_interconnect_m_axi_bresp      [3:2]), // output
    // .s_axi_bvalid                                       (axil_interconnect_m_axi_bvalid     [1:1]), // output
    // .s_axi_bready                                       (axil_interconnect_m_axi_bready     [1:1]), // input

    // Node Scoreboard -> Transformation Engine Interface
    .nsb_fte_req_valid                                  (nsb_fte_req_valid),
    .nsb_fte_req_ready                                  (nsb_fte_req_ready),
    .nsb_fte_req                                        (nsb_fte_req),
    .nsb_fte_resp_valid                                 (nsb_fte_resp_valid),
    .nsb_fte_resp                                       (nsb_fte_resp),

    .feature_channel_req_valid                           (feature_channel_req_valid),
    .feature_channel_req_ready                           (feature_channel_req_ready),
    .feature_channel_req                                 (feature_channel_req),
    .feature_channel_resp_valid                          (feature_channel_resp_valid),
    .feature_channel_resp_ready                          (feature_channel_resp_ready),
    .feature_channel_resp                                (feature_channel_resp),

    .weight_channel_req_valid                           (weight_channel_req_valid),
    .weight_channel_req_ready                           (weight_channel_req_ready),
    .weight_channel_req                                 (weight_channel_req),
    .weight_channel_resp_valid                          (weight_channel_resp_valid),
    .weight_channel_resp_ready                          (weight_channel_resp_ready),
    .weight_channel_resp                                (weight_channel_resp),

    .transformation_engine_axi_interconnect_axi_araddr,
    .transformation_engine_axi_interconnect_axi_arburst,
    .transformation_engine_axi_interconnect_axi_arcache,
    .transformation_engine_axi_interconnect_axi_arid,
    .transformation_engine_axi_interconnect_axi_arlen,
    .transformation_engine_axi_interconnect_axi_arlock,
    .transformation_engine_axi_interconnect_axi_arprot,
    .transformation_engine_axi_interconnect_axi_arqos,
    .transformation_engine_axi_interconnect_axi_arsize,
    .transformation_engine_axi_interconnect_axi_arvalid,
    .transformation_engine_axi_interconnect_axi_arready,
    
    .transformation_engine_axi_interconnect_axi_awaddr,
    .transformation_engine_axi_interconnect_axi_awburst,
    .transformation_engine_axi_interconnect_axi_awcache,
    .transformation_engine_axi_interconnect_axi_awid,
    .transformation_engine_axi_interconnect_axi_awlen,
    .transformation_engine_axi_interconnect_axi_awlock,
    .transformation_engine_axi_interconnect_axi_awprot,
    .transformation_engine_axi_interconnect_axi_awqos,
    .transformation_engine_axi_interconnect_axi_awready,
    .transformation_engine_axi_interconnect_axi_awsize,
    .transformation_engine_axi_interconnect_axi_awvalid,
    
    .transformation_engine_axi_interconnect_axi_bid,
    .transformation_engine_axi_interconnect_axi_bready,
    .transformation_engine_axi_interconnect_axi_bresp,
    .transformation_engine_axi_interconnect_axi_bvalid,
    
    .transformation_engine_axi_interconnect_axi_rdata,
    .transformation_engine_axi_interconnect_axi_rid,
    .transformation_engine_axi_interconnect_axi_rlast,
    .transformation_engine_axi_interconnect_axi_rready,
    .transformation_engine_axi_interconnect_axi_rresp,
    .transformation_engine_axi_interconnect_axi_rvalid,
    
    .transformation_engine_axi_interconnect_axi_wdata,
    .transformation_engine_axi_interconnect_axi_wlast,
    .transformation_engine_axi_interconnect_axi_wready,
    .transformation_engine_axi_interconnect_axi_wstrb,
    .transformation_engine_axi_interconnect_axi_wvalid,

    // Layer configuration
    .layer_config_in_features_count,
    .layer_config_out_features_count,                                   
    .layer_config_activation_function_value,
    .layer_config_bias_value,
    .layer_config_leaky_relu_alpha_value,
    .layer_config_out_features_address_msb_value,
    .layer_config_out_features_address_lsb_value,
    .ctrl_buffering_enable_value,
    .ctrl_writeback_enable_value
);


// ====================================================================================
// Interconnect
// ====================================================================================


// ====================================================================================
// AXI Memory Interconnect
// ====================================================================================

// S00: Prefetcher (Feature bank) ---------- / read-only
// S01: Prefetcher (weight bank) ----------- / read-only
// S02: top (TB) --------------------------- / read-write
// S03: FTE -------------------------------- / write-only
// S04: Prefetcher (message read master) --- / read-only

// S05: unused ----------------------------- / read-write
// S06: unused ----------------------------- / read-write
// S07: unused ----------------------------- / read-write

axi_interconnect_0 axi_memory_interconnect_i (
    .INTERCONNECT_ACLK            (clk),        // input wire INTERCONNECT_ACLK
    .INTERCONNECT_ARESETN         (!rst),       // input wire INTERCONNECT_ARESETN

    // S00: unused
    .S00_AXI_ACLK                 (clk), // input wire S00_AXI_ACLK
    .S00_AXI_ARESET_OUT_N         (S00_AXI_ARESET_OUT_N),   // output wire S00_AXI_ARESET_OUT_N
    .S00_AXI_ARADDR               ('0), // input wire [33 : 0] S00_AXI_ARADDR
    .S00_AXI_ARBURST              ('0), // input wire [1 : 0] S00_AXI_ARBURST
    .S00_AXI_ARCACHE              ('0), // input wire [3 : 0] S00_AXI_ARCACHE
    .S00_AXI_ARID                 ('0), // input wire [0 : 0] S00_AXI_ARID
    .S00_AXI_ARLEN                ('0), // input wire [7 : 0] S00_AXI_ARLEN
    .S00_AXI_ARLOCK               ('0), // input wire S00_AXI_ARLOCK
    .S00_AXI_ARPROT               ('0), // input wire [2 : 0] S00_AXI_ARPROT
    .S00_AXI_ARQOS                ('0), // input wire [3 : 0] S00_AXI_ARQOS
    .S00_AXI_ARSIZE               ('0), // input wire [2 : 0] S00_AXI_ARSIZE
    .S00_AXI_ARVALID              ('0), // input wire S00_AXI_ARVALID
    .S00_AXI_ARREADY              (),   // output wire S00_AXI_ARREADY
    .S00_AXI_AWADDR               ('0), // input wire [33 : 0] S00_AXI_AWADDR
    .S00_AXI_AWBURST              ('0), // input wire [1 : 0] S00_AXI_AWBURST
    .S00_AXI_AWCACHE              ('0), // input wire [3 : 0] S00_AXI_AWCACHE
    .S00_AXI_AWID                 ('0), // input wire [0 : 0] S00_AXI_AWID
    .S00_AXI_AWLEN                ('0), // input wire [7 : 0] S00_AXI_AWLEN
    .S00_AXI_AWLOCK               ('0), // input wire S00_AXI_AWLOCK
    .S00_AXI_AWPROT               ('0), // input wire [2 : 0] S00_AXI_AWPROT
    .S00_AXI_AWQOS                ('0), // input wire [3 : 0] S00_AXI_AWQOS
    .S00_AXI_AWREADY              (),   // output wire S00_AXI_AWREADY
    .S00_AXI_AWSIZE               ('0), // input wire [2 : 0] S00_AXI_AWSIZE
    .S00_AXI_AWVALID              ('0), // input wire S00_AXI_AWVALID
    .S00_AXI_BID                  (),   // output wire [0 : 0] S00_AXI_BID
    .S00_AXI_BREADY               ('0), // input wire S00_AXI_BREADY
    .S00_AXI_BRESP                (),   // output wire [1 : 0] S00_AXI_BRESP
    .S00_AXI_BVALID               (),   // output wire S00_AXI_BVALID
    .S00_AXI_RDATA                (),   // output wire [511 : 0] S00_AXI_RDATA
    .S00_AXI_RID                  (),   // output wire [0 : 0] S00_AXI_RID
    .S00_AXI_RLAST                (),   // output wire S00_AXI_RLAST
    .S00_AXI_RREADY               ('0), // input wire S00_AXI_RREADY
    .S00_AXI_RRESP                (),   // output wire [1 : 0] S00_AXI_RRESP
    .S00_AXI_RVALID               (),   // output wire S00_AXI_RVALID
    .S00_AXI_WDATA                ('0), // input wire [511 : 0] S00_AXI_WDATA
    .S00_AXI_WLAST                ('0), // input wire S00_AXI_WLAST
    .S00_AXI_WREADY               (),   // output wire S00_AXI_WREADY
    .S00_AXI_WSTRB                ('0), // input wire [63 : 0] S00_AXI_WSTRB
    .S00_AXI_WVALID               ('0), // input wire S00_AXI_WVALID

    // // S00: Prefetcher Feature Bank 
    // .S00_AXI_ACLK                 (clk),                  // input wire S00_AXI_ACLK
    // .S00_AXI_ARESET_OUT_N         (S00_AXI_ARESET_OUT_N),  // output wire S00_AXI_ARESET_OUT_N


    // .S00_AXI_ARADDR               (prefetcher_feature_bank_rm_axi_interconnect_axi_araddr   ),          // input wire [33 : 0] S00_AXI_ARADDR
    // .S00_AXI_ARBURST              (prefetcher_feature_bank_rm_axi_interconnect_axi_arburst  ),            // input wire [1 : 0] S00_AXI_ARBURST
    // .S00_AXI_ARCACHE              (prefetcher_feature_bank_rm_axi_interconnect_axi_arcache  ),            // input wire [3 : 0] S00_AXI_ARCACHE
    // .S00_AXI_ARID                 (prefetcher_feature_bank_rm_axi_interconnect_axi_arid     ),              // input wire [0 : 0] S00_AXI_ARID
    // .S00_AXI_ARLEN                (prefetcher_feature_bank_rm_axi_interconnect_axi_arlen    ),            // input wire [7 : 0] S00_AXI_ARLEN
    // .S00_AXI_ARLOCK               (prefetcher_feature_bank_rm_axi_interconnect_axi_arlock   ),                           // input wire S00_AXI_ARLOCK
    // .S00_AXI_ARPROT               (prefetcher_feature_bank_rm_axi_interconnect_axi_arprot   ),                           // input wire [2 : 0] S00_AXI_ARPROT
    // .S00_AXI_ARQOS                (prefetcher_feature_bank_rm_axi_interconnect_axi_arqos    ),                           // input wire [3 : 0] S00_AXI_ARQOS
    // .S00_AXI_ARSIZE               (prefetcher_feature_bank_rm_axi_interconnect_axi_arsize   ),          // input wire [2 : 0] S00_AXI_ARSIZE
    // .S00_AXI_ARVALID              (prefetcher_feature_bank_rm_axi_interconnect_axi_arvalid  ),            // input wire S00_AXI_ARVALID
    // .S00_AXI_ARREADY              (prefetcher_feature_bank_rm_axi_interconnect_axi_arready  ),            // output wire S00_AXI_ARREADY
    // .S00_AXI_AWADDR               (prefetcher_feature_bank_rm_axi_interconnect_axi_awaddr   ),          // input wire [33 : 0] S00_AXI_AWADDR
    // .S00_AXI_AWBURST              (prefetcher_feature_bank_rm_axi_interconnect_axi_awburst  ),            // input wire [1 : 0] S00_AXI_AWBURST
    // .S00_AXI_AWCACHE              (prefetcher_feature_bank_rm_axi_interconnect_axi_awcache  ),            // input wire [3 : 0] S00_AXI_AWCACHE
    // .S00_AXI_AWID                 (prefetcher_feature_bank_rm_axi_interconnect_axi_awid     ),              // input wire [0 : 0] S00_AXI_AWID
    // .S00_AXI_AWLEN                (prefetcher_feature_bank_rm_axi_interconnect_axi_awlen    ),            // input wire [7 : 0] S00_AXI_AWLEN
    // .S00_AXI_AWLOCK               (prefetcher_feature_bank_rm_axi_interconnect_axi_awlock   ),                           // input wire S00_AXI_AWLOCK
    // .S00_AXI_AWPROT               (prefetcher_feature_bank_rm_axi_interconnect_axi_awprot   ),          // input wire [2 : 0] S00_AXI_AWPROT
    // .S00_AXI_AWQOS                (prefetcher_feature_bank_rm_axi_interconnect_axi_awqos    ),                           // input wire [3 : 0] S00_AXI_AWQOS
    // .S00_AXI_AWREADY              (prefetcher_feature_bank_rm_axi_interconnect_axi_awready  ),            // output wire S00_AXI_AWREADY
    // .S00_AXI_AWSIZE               (prefetcher_feature_bank_rm_axi_interconnect_axi_awsize   ),          // input wire [2 : 0] S00_AXI_AWSIZE
    // .S00_AXI_AWVALID              (prefetcher_feature_bank_rm_axi_interconnect_axi_awvalid  ),            // input wire S00_AXI_AWVALID
    // .S00_AXI_BID                  (prefetcher_feature_bank_rm_axi_interconnect_axi_bid      ),                // output wire [0 : 0] S00_AXI_BID
    // .S00_AXI_BREADY               (prefetcher_feature_bank_rm_axi_interconnect_axi_bready   ),          // input wire S00_AXI_BREADY
    // .S00_AXI_BRESP                (prefetcher_feature_bank_rm_axi_interconnect_axi_bresp    ),            // output wire [1 : 0] S00_AXI_BRESP
    // .S00_AXI_BVALID               (prefetcher_feature_bank_rm_axi_interconnect_axi_bvalid   ),          // output wire S00_AXI_BVALID
    // .S00_AXI_RDATA                (prefetcher_feature_bank_rm_axi_interconnect_axi_rdata    ),            // output wire [511 : 0] S00_AXI_RDATA
    // .S00_AXI_RID                  (prefetcher_feature_bank_rm_axi_interconnect_axi_rid      ),                // output wire [0 : 0] S00_AXI_RID
    // .S00_AXI_RLAST                (prefetcher_feature_bank_rm_axi_interconnect_axi_rlast    ),            // output wire S00_AXI_RLAST
    // .S00_AXI_RREADY               (prefetcher_feature_bank_rm_axi_interconnect_axi_rready   ),          // input wire S00_AXI_RREADY
    // .S00_AXI_RRESP                (prefetcher_feature_bank_rm_axi_interconnect_axi_rresp    ),            // output wire [1 : 0] S00_AXI_RRESP
    // .S00_AXI_RVALID               (prefetcher_feature_bank_rm_axi_interconnect_axi_rvalid   ),          // output wire S00_AXI_RVALID
    // .S00_AXI_WDATA                (prefetcher_feature_bank_rm_axi_interconnect_axi_wdata    ),            // input wire [511 : 0] S00_AXI_WDATA
    // .S00_AXI_WLAST                (prefetcher_feature_bank_rm_axi_interconnect_axi_wlast    ),            // input wire S00_AXI_WLAST
    // .S00_AXI_WREADY               (prefetcher_feature_bank_rm_axi_interconnect_axi_wready   ),          // output wire S00_AXI_WREADY
    // .S00_AXI_WSTRB                (prefetcher_feature_bank_rm_axi_interconnect_axi_wstrb    ),            // input wire [63 : 0] S00_AXI_WSTRB
    // .S00_AXI_WVALID               (prefetcher_feature_bank_rm_axi_interconnect_axi_wvalid   ),          // input wire S00_AXI_WVALID


    // S01: Prefetcher Weight Bank
    .S01_AXI_ACLK                 (clk),                  // input wire S01_AXI_ACLK
    .S01_AXI_ARESET_OUT_N         (S01_AXI_ARESET_OUT_N),  // output wire S01_AXI_ARESET_OUT_N

    .S01_AXI_ARADDR               (prefetcher_weight_bank_rm_axi_interconnect_axi_araddr   ),
    .S01_AXI_ARBURST              (prefetcher_weight_bank_rm_axi_interconnect_axi_arburst  ),
    .S01_AXI_ARCACHE              (prefetcher_weight_bank_rm_axi_interconnect_axi_arcache  ),
    .S01_AXI_ARID                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arid     ),
    .S01_AXI_ARLEN                (prefetcher_weight_bank_rm_axi_interconnect_axi_arlen    ),
    .S01_AXI_ARLOCK               (prefetcher_weight_bank_rm_axi_interconnect_axi_arlock   ),
    .S01_AXI_ARPROT               (prefetcher_weight_bank_rm_axi_interconnect_axi_arprot   ),
    .S01_AXI_ARQOS                (prefetcher_weight_bank_rm_axi_interconnect_axi_arqos    ),
    .S01_AXI_ARSIZE               (prefetcher_weight_bank_rm_axi_interconnect_axi_arsize   ),
    .S01_AXI_ARVALID              (prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid  ),
    .S01_AXI_ARREADY              (prefetcher_weight_bank_rm_axi_interconnect_axi_arready  ),
    .S01_AXI_AWADDR               (prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr   ),
    .S01_AXI_AWBURST              (prefetcher_weight_bank_rm_axi_interconnect_axi_awburst  ),
    .S01_AXI_AWCACHE              (prefetcher_weight_bank_rm_axi_interconnect_axi_awcache  ),
    .S01_AXI_AWID                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awid     ),
    .S01_AXI_AWLEN                (prefetcher_weight_bank_rm_axi_interconnect_axi_awlen    ),
    .S01_AXI_AWLOCK               (prefetcher_weight_bank_rm_axi_interconnect_axi_awlock   ),
    .S01_AXI_AWPROT               (prefetcher_weight_bank_rm_axi_interconnect_axi_awprot   ),
    .S01_AXI_AWQOS                (prefetcher_weight_bank_rm_axi_interconnect_axi_awqos    ),
    .S01_AXI_AWREADY              (prefetcher_weight_bank_rm_axi_interconnect_axi_awready  ),
    .S01_AXI_AWSIZE               (prefetcher_weight_bank_rm_axi_interconnect_axi_awsize   ),
    .S01_AXI_AWVALID              (prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid  ),
    .S01_AXI_BID                  (prefetcher_weight_bank_rm_axi_interconnect_axi_bid      ),
    .S01_AXI_BREADY               (prefetcher_weight_bank_rm_axi_interconnect_axi_bready   ),
    .S01_AXI_BRESP                (prefetcher_weight_bank_rm_axi_interconnect_axi_bresp    ),
    .S01_AXI_BVALID               (prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid   ),
    .S01_AXI_RDATA                (prefetcher_weight_bank_rm_axi_interconnect_axi_rdata    ),
    .S01_AXI_RID                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rid      ),
    .S01_AXI_RLAST                (prefetcher_weight_bank_rm_axi_interconnect_axi_rlast    ),
    .S01_AXI_RREADY               (prefetcher_weight_bank_rm_axi_interconnect_axi_rready   ),
    .S01_AXI_RRESP                (prefetcher_weight_bank_rm_axi_interconnect_axi_rresp    ),
    .S01_AXI_RVALID               (prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid   ),
    .S01_AXI_WDATA                (prefetcher_weight_bank_rm_axi_interconnect_axi_wdata    ),
    .S01_AXI_WLAST                (prefetcher_weight_bank_rm_axi_interconnect_axi_wlast    ),
    .S01_AXI_WREADY               (prefetcher_weight_bank_rm_axi_interconnect_axi_wready   ),
    .S01_AXI_WSTRB                (prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb    ),
    .S01_AXI_WVALID               (prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid   ),

    // S02: FTE
    // .S02_AXI_ACLK                 (sys_clk),                  // input wire S02_AXI_ACLK
    // .S02_AXI_ARESET_OUT_N         (S02_AXI_ARESET_OUT_N),  // output wire S02_AXI_ARESET_OUT_N

    // .S02_AXI_AWID                 (transformation_engine_axi_interconnect_axi_awid),                  // input wire [0 : 0] S02_AXI_AWID
    // .S02_AXI_AWADDR               (transformation_engine_axi_interconnect_axi_awaddr),              // input wire [33 : 0] S02_AXI_AWADDR
    // .S02_AXI_AWLEN                (transformation_engine_axi_interconnect_axi_awlen),                // input wire [7 : 0] S02_AXI_AWLEN
    // .S02_AXI_AWSIZE               (transformation_engine_axi_interconnect_axi_awsize),              // input wire [2 : 0] S02_AXI_AWSIZE
    // .S02_AXI_AWBURST              (transformation_engine_axi_interconnect_axi_awburst),            // input wire [1 : 0] S02_AXI_AWBURST
    // .S02_AXI_AWLOCK               (transformation_engine_axi_interconnect_axi_awlock),              // input wire S02_AXI_AWLOCK
    // .S02_AXI_AWCACHE              (transformation_engine_axi_interconnect_axi_awcache),            // input wire [3 : 0] S02_AXI_AWCACHE
    // .S02_AXI_AWPROT               (transformation_engine_axi_interconnect_axi_awprot),              // input wire [2 : 0] S02_AXI_AWPROT
    // .S02_AXI_AWQOS                (transformation_engine_axi_interconnect_axi_awqos),                // input wire [3 : 0] S02_AXI_AWQOS
    // .S02_AXI_AWVALID              (transformation_engine_axi_interconnect_axi_awvalid),            // input wire S02_AXI_AWVALID
    // .S02_AXI_AWREADY              (transformation_engine_axi_interconnect_axi_awready),            // output wire S02_AXI_AWREADY
    // .S02_AXI_WDATA                (transformation_engine_axi_interconnect_axi_wdata),                // input wire [511 : 0] S02_AXI_WDATA
    // .S02_AXI_WSTRB                (transformation_engine_axi_interconnect_axi_wstrb),                // input wire [63 : 0] S02_AXI_WSTRB
    // .S02_AXI_WLAST                (transformation_engine_axi_interconnect_axi_wlast),                // input wire S02_AXI_WLAST
    // .S02_AXI_WVALID               (transformation_engine_axi_interconnect_axi_wvalid),              // input wire S02_AXI_WVALID
    // .S02_AXI_WREADY               (transformation_engine_axi_interconnect_axi_wready),              // output wire S02_AXI_WREADY
    // .S02_AXI_BID                  (transformation_engine_axi_interconnect_axi_bid),                    // output wire [0 : 0] S02_AXI_BID
    // .S02_AXI_BRESP                (transformation_engine_axi_interconnect_axi_bresp),                // output wire [1 : 0] S02_AXI_BRESP
    // .S02_AXI_BVALID               (transformation_engine_axi_interconnect_axi_bvalid),              // output wire S02_AXI_BVALID
    // .S02_AXI_BREADY               (transformation_engine_axi_interconnect_axi_bready),              // input wire S02_AXI_BREADY
    // .S02_AXI_ARID                 (transformation_engine_axi_interconnect_axi_arid),                  // input wire [0 : 0] S02_AXI_ARID
    // .S02_AXI_ARADDR               (transformation_engine_axi_interconnect_axi_araddr),              // input wire [33 : 0] S02_AXI_ARADDR
    // .S02_AXI_ARLEN                (transformation_engine_axi_interconnect_axi_arlen),                // input wire [7 : 0] S02_AXI_ARLEN
    // .S02_AXI_ARSIZE               (transformation_engine_axi_interconnect_axi_arsize),              // input wire [2 : 0] S02_AXI_ARSIZE
    // .S02_AXI_ARBURST              (transformation_engine_axi_interconnect_axi_arburst),            // input wire [1 : 0] S02_AXI_ARBURST
    // .S02_AXI_ARLOCK               (transformation_engine_axi_interconnect_axi_arlock),              // input wire S02_AXI_ARLOCK
    // .S02_AXI_ARCACHE              (transformation_engine_axi_interconnect_axi_arcache),            // input wire [3 : 0] S02_AXI_ARCACHE
    // .S02_AXI_ARPROT               (transformation_engine_axi_interconnect_axi_arprot),              // input wire [2 : 0] S02_AXI_ARPROT
    // .S02_AXI_ARQOS                (transformation_engine_axi_interconnect_axi_arqos),                // input wire [3 : 0] S02_AXI_ARQOS
    // .S02_AXI_ARVALID              (transformation_engine_axi_interconnect_axi_arvalid),            // input wire S02_AXI_ARVALID
    // .S02_AXI_ARREADY              (transformation_engine_axi_interconnect_axi_arready),            // output wire S02_AXI_ARREADY
    // .S02_AXI_RID                  (transformation_engine_axi_interconnect_axi_rid),                    // output wire [0 : 0] S02_AXI_RID
    // .S02_AXI_RDATA                (transformation_engine_axi_interconnect_axi_rdata),                // output wire [511 : 0] S02_AXI_RDATA
    // .S02_AXI_RRESP                (transformation_engine_axi_interconnect_axi_rresp),                // output wire [1 : 0] S02_AXI_RRESP
    // .S02_AXI_RLAST                (transformation_engine_axi_interconnect_axi_rlast),                // output wire S02_AXI_RLAST
    // .S02_AXI_RVALID               (transformation_engine_axi_interconnect_axi_rvalid),              // output wire S02_AXI_RVALID
    // .S02_AXI_RREADY               (transformation_engine_axi_interconnect_axi_rready),              // input wire S02_AXI_RREADY2

    // S02: unused
    .S02_AXI_ACLK                 (clk), // input wire S00_AXI_ACLK
    .S02_AXI_ARESET_OUT_N         (S02_AXI_ARESET_OUT_N),   // output wire S00_AXI_ARESET_OUT_N
    .S02_AXI_ARADDR               ('0), // input wire [33 : 0] S00_AXI_ARADDR
    .S02_AXI_ARBURST              ('0), // input wire [1 : 0] S00_AXI_ARBURST
    .S02_AXI_ARCACHE              ('0), // input wire [3 : 0] S00_AXI_ARCACHE
    .S02_AXI_ARID                 ('0), // input wire [0 : 0] S00_AXI_ARID
    .S02_AXI_ARLEN                ('0), // input wire [7 : 0] S00_AXI_ARLEN
    .S02_AXI_ARLOCK               ('0), // input wire S00_AXI_ARLOCK
    .S02_AXI_ARPROT               ('0), // input wire [2 : 0] S00_AXI_ARPROT
    .S02_AXI_ARQOS                ('0), // input wire [3 : 0] S00_AXI_ARQOS
    .S02_AXI_ARSIZE               ('0), // input wire [2 : 0] S00_AXI_ARSIZE
    .S02_AXI_ARVALID              ('0), // input wire S00_AXI_ARVALID
    .S02_AXI_ARREADY              (),   // output wire S00_AXI_ARREADY
    .S02_AXI_AWADDR               ('0), // input wire [33 : 0] S00_AXI_AWADDR
    .S02_AXI_AWBURST              ('0), // input wire [1 : 0] S00_AXI_AWBURST
    .S02_AXI_AWCACHE              ('0), // input wire [3 : 0] S00_AXI_AWCACHE
    .S02_AXI_AWID                 ('0), // input wire [0 : 0] S00_AXI_AWID
    .S02_AXI_AWLEN                ('0), // input wire [7 : 0] S00_AXI_AWLEN
    .S02_AXI_AWLOCK               ('0), // input wire S00_AXI_AWLOCK
    .S02_AXI_AWPROT               ('0), // input wire [2 : 0] S00_AXI_AWPROT
    .S02_AXI_AWQOS                ('0), // input wire [3 : 0] S00_AXI_AWQOS
    .S02_AXI_AWREADY              (),   // output wire S00_AXI_AWREADY
    .S02_AXI_AWSIZE               ('0), // input wire [2 : 0] S00_AXI_AWSIZE
    .S02_AXI_AWVALID              ('0), // input wire S00_AXI_AWVALID
    .S02_AXI_BID                  (),   // output wire [0 : 0] S00_AXI_BID
    .S02_AXI_BREADY               ('0), // input wire S00_AXI_BREADY
    .S02_AXI_BRESP                (),   // output wire [1 : 0] S00_AXI_BRESP
    .S02_AXI_BVALID               (),   // output wire S00_AXI_BVALID
    .S02_AXI_RDATA                (),   // output wire [511 : 0] S00_AXI_RDATA
    .S02_AXI_RID                  (),   // output wire [0 : 0] S00_AXI_RID
    .S02_AXI_RLAST                (),   // output wire S00_AXI_RLAST
    .S02_AXI_RREADY               ('0), // input wire S00_AXI_RREADY
    .S02_AXI_RRESP                (),   // output wire [1 : 0] S00_AXI_RRESP
    .S02_AXI_RVALID               (),   // output wire S00_AXI_RVALID
    .S02_AXI_WDATA                ('0), // input wire [511 : 0] S00_AXI_WDATA
    .S02_AXI_WLAST                ('0), // input wire S00_AXI_WLAST
    .S02_AXI_WREADY               (),   // output wire S00_AXI_WREADY
    .S02_AXI_WSTRB                ('0), // input wire [63 : 0] S00_AXI_WSTRB
    .S02_AXI_WVALID               ('0), // input wire S00_AXI_WVALID


    
    // M00: DDR4 controller or RAM model (depending on DRAM_CONTROLLER and RAM_MODEL macros)
    .M00_AXI_ACLK                 (clk),                  // input wire M00_AXI_ACLK
    .M00_AXI_ARESET_OUT_N         (M00_AXI_ARESET_OUT_N),  // output wire M00_AXI_ARESET_OUT_N

    .M00_AXI_AWID                 (c0_ddr4_s_axi_awid),                  // output wire [3 : 0] M00_AXI_AWID
    .M00_AXI_AWADDR               (c0_ddr4_s_axi_awaddr),              // output wire [33 : 0] M00_AXI_AWADDR
    .M00_AXI_AWLEN                (c0_ddr4_s_axi_awlen),                // output wire [7 : 0] M00_AXI_AWLEN
    .M00_AXI_AWSIZE               (c0_ddr4_s_axi_awsize),              // output wire [2 : 0] M00_AXI_AWSIZE
    .M00_AXI_AWBURST              (c0_ddr4_s_axi_awburst),            // output wire [1 : 0] M00_AXI_AWBURST
    .M00_AXI_AWLOCK               (c0_ddr4_s_axi_awlock),              // output wire M00_AXI_AWLOCK
    .M00_AXI_AWCACHE              (c0_ddr4_s_axi_awcache),            // output wire [3 : 0] M00_AXI_AWCACHE
    .M00_AXI_AWPROT               (c0_ddr4_s_axi_awprot),              // output wire [2 : 0] M00_AXI_AWPROT
    .M00_AXI_AWQOS                (c0_ddr4_s_axi_awqos),                // output wire [3 : 0] M00_AXI_AWQOS
    .M00_AXI_AWVALID              (c0_ddr4_s_axi_awvalid),            // output wire M00_AXI_AWVALID
    .M00_AXI_AWREADY              (c0_ddr4_s_axi_awready),            // input wire M00_AXI_AWREADY
    .M00_AXI_WDATA                (c0_ddr4_s_axi_wdata),                // output wire [511 : 0] M00_AXI_WDATA
    .M00_AXI_WSTRB                (c0_ddr4_s_axi_wstrb),                // output wire [63 : 0] M00_AXI_WSTRB
    .M00_AXI_WLAST                (c0_ddr4_s_axi_wlast),                // output wire M00_AXI_WLAST
    .M00_AXI_WVALID               (c0_ddr4_s_axi_wvalid),              // output wire M00_AXI_WVALID
    .M00_AXI_WREADY               (c0_ddr4_s_axi_wready),              // input wire M00_AXI_WREADY
    .M00_AXI_BREADY               (c0_ddr4_s_axi_bready),              // output wire M00_AXI_BREADY
    .M00_AXI_BID                  (c0_ddr4_s_axi_bid),                    // input wire [3 : 0] M00_AXI_BID
    .M00_AXI_BRESP                (c0_ddr4_s_axi_bresp),                // input wire [1 : 0] M00_AXI_BRESP
    .M00_AXI_BVALID               (c0_ddr4_s_axi_bvalid),              // input wire M00_AXI_BVALID
    .M00_AXI_ARID                 (c0_ddr4_s_axi_arid),                  // output wire [3 : 0] M00_AXI_ARID
    .M00_AXI_ARADDR               (c0_ddr4_s_axi_araddr),              // output wire [33 : 0] M00_AXI_ARADDR
    .M00_AXI_ARLEN                (c0_ddr4_s_axi_arlen),                // output wire [7 : 0] M00_AXI_ARLEN
    .M00_AXI_ARSIZE               (c0_ddr4_s_axi_arsize),              // output wire [2 : 0] M00_AXI_ARSIZE
    .M00_AXI_ARBURST              (c0_ddr4_s_axi_arburst),            // output wire [1 : 0] M00_AXI_ARBURST
    .M00_AXI_ARLOCK               (c0_ddr4_s_axi_arlock),              // output wire M00_AXI_ARLOCK
    .M00_AXI_ARCACHE              (c0_ddr4_s_axi_arcache),            // output wire [3 : 0] M00_AXI_ARCACHE
    .M00_AXI_ARPROT               (c0_ddr4_s_axi_arprot),              // output wire [2 : 0] M00_AXI_ARPROT
    .M00_AXI_ARQOS                (c0_ddr4_s_axi_arqos),                // output wire [3 : 0] M00_AXI_ARQOS
    .M00_AXI_ARVALID              (c0_ddr4_s_axi_arvalid),            // output wire M00_AXI_ARVALID
    .M00_AXI_ARREADY              (c0_ddr4_s_axi_arready),            // input wire M00_AXI_ARREADY
    .M00_AXI_RREADY               (c0_ddr4_s_axi_rready),              // output wire M00_AXI_RREADY
    .M00_AXI_RID                  (c0_ddr4_s_axi_rid),                    // input wire [3 : 0] M00_AXI_RID
    .M00_AXI_RDATA                (c0_ddr4_s_axi_rdata),                // input wire [511 : 0] M00_AXI_RDATA
    .M00_AXI_RRESP                (c0_ddr4_s_axi_rresp),                // input wire [1 : 0] M00_AXI_RRESP
    .M00_AXI_RLAST                (c0_ddr4_s_axi_rlast),                // input wire M00_AXI_RLAST
    .M00_AXI_RVALID               (c0_ddr4_s_axi_rvalid)              // input wire M00_AXI_RVALID
);



endmodule
