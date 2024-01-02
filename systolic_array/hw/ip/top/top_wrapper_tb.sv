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

    input logic                           sys_clk,
    input logic                           sys_rst, //Common port for all controllers

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
    // output logic                          c0_ddr4_s_axi_rready

    // nsb -> prefetcher
    input   logic                                                nsb_prefetcher_req_valid,
    output  logic                                                nsb_prefetcher_req_ready,
    input   NSB_PREF_REQ_t                                       nsb_prefetcher_req,
    output  logic                                                nsb_prefetcher_resp_valid,
    output  NSB_PREF_RESP_t                                      nsb_prefetcher_resp
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

// Weight Channel: FTE -> Prefetcher
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_req_valid;
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_req_ready;
WEIGHT_CHANNEL_REQ_t  [top_pkg::PRECISION_COUNT-1:0] weight_channel_req;
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp_valid;
logic                 [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp_ready;
WEIGHT_CHANNEL_RESP_t [top_pkg::PRECISION_COUNT-1:0] weight_channel_resp;

// ====================================================================================
// AXI Memory 
// ====================================================================================
// axi_ram #(
//     // Width of data bus in bits
//     .DATA_WIDTH (32),
//     // Width of address bus in bits
//     .ADDR_WIDTH (16),
//     // Width of wstrb (width of data bus in words)
//     .STRB_WIDTH  (32/8),
//     // Width of ID signal
//     .ID_WIDTH        (8),
//     // Extra pipeline register on output
//     .PIPELINE_OUTPUT (0),
// ) axi_ram (
//     .clk                        (clk),
//     .rst                        (rst),

//     .s_axi_awid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awid),
//     .s_axi_awaddr               (prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr),
//     .s_axi_awlen                (prefetcher_weight_bank_rm_axi_interconnect_axi_awlen),
//     .s_axi_awsize               (prefetcher_weight_bank_rm_axi_interconnect_axi_awsize),
//     .s_axi_awburst              (prefetcher_weight_bank_rm_axi_interconnect_axi_awburst),
//     .s_axi_awlock               (prefetcher_weight_bank_rm_axi_interconnect_axi_awlock),
//     .s_axi_awcache              (prefetcher_weight_bank_rm_axi_interconnect_axi_awcache), 
//     .s_axi_awprot               (prefetcher_weight_bank_rm_axi_interconnect_axi_awprot),
//     .s_axi_awvalid              (prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid),
//     .s_axi_awready              (prefetcher_weight_bank_rm_axi_interconnect_axi_awready),
//     .s_axi_wdata                (prefetcher_weight_bank_rm_axi_interconnect_axi_wdata),
//     .s_axi_wstrb                (prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb),
//     .s_axi_wlast                (prefetcher_weight_bank_rm_axi_interconnect_axi_wlast),
//     .s_axi_wvalid               (prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid),
//     .s_axi_wready               (prefetcher_weight_bank_rm_axi_interconnect_axi_wready),
//     .s_axi_bid                  (prefetcher_weight_bank_rm_axi_interconnect_axi_bid),
//     .s_axi_bresp                (prefetcher_weight_bank_rm_axi_interconnect_axi_bresp),
//     .s_axi_bvalid               (prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid),
//     .s_axi_bready               (prefetcher_weight_bank_rm_axi_interconnect_axi_bready),
//     .s_axi_arid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arid),
//     .s_axi_araddr               (prefetcher_weight_bank_rm_axi_interconnect_axi_araddr),
//     .s_axi_arlen                (prefetcher_weight_bank_rm_axi_interconnect_axi_arlen),
//     .s_axi_arsize               (prefetcher_weight_bank_rm_axi_interconnect_axi_arsize),
//     .s_axi_arburst              (prefetcher_weight_bank_rm_axi_interconnect_axi_arburst),
//     .s_axi_arlock               (prefetcher_weight_bank_rm_axi_interconnect_axi_arlock),
//     .s_axi_arcache              (prefetcher_weight_bank_rm_axi_interconnect_axi_arcache),
//     .s_axi_arprot               (prefetcher_weight_bank_rm_axi_interconnect_axi_arprot),
//     .s_axi_arvalid              (prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid),
//     .s_axi_arready              (prefetcher_weight_bank_rm_axi_interconnect_axi_arready),
//     .s_axi_rid                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rid),
//     .s_axi_rdata                (prefetcher_weight_bank_rm_axi_interconnect_axi_rdata),
//     .s_axi_rresp                (prefetcher_weight_bank_rm_axi_interconnect_axi_rresp),
//     .s_axi_rlast                (prefetcher_weight_bank_rm_axi_interconnect_axi_rlast),
//     .s_axi_rvalid               (prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid),
//     .s_axi_rready               (prefetcher_weight_bank_rm_axi_interconnect_axi_rready)
// );

axi_interface axi_ram (
    .clk                        (clk),
    .rst                        (rst),

    .axi_awid                   (prefetcher_weight_bank_rm_axi_interconnect_axi_awid),
    .axi_awaddr                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr),
    .axi_awlen                  (prefetcher_weight_bank_rm_axi_interconnect_axi_awlen),
    .axi_awsize                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awsize),
    .axi_awburst                (prefetcher_weight_bank_rm_axi_interconnect_axi_awburst),
    .axi_awlock                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awlock),
    .axi_awcache                (prefetcher_weight_bank_rm_axi_interconnect_axi_awcache),
    .axi_awprot                 (prefetcher_weight_bank_rm_axi_interconnect_axi_awprot),
    .axi_awqos                  (prefetcher_weight_bank_rm_axi_interconnect_axi_awqos),
    .axi_awregion               (), // not used
    .axi_awvalid                (prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid),
    .axi_awready                (prefetcher_weight_bank_rm_axi_interconnect_axi_awready),
    .axi_wdata                  (prefetcher_weight_bank_rm_axi_interconnect_axi_wdata),
    .axi_wstrb                  (prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb),
    .axi_wlast                  (prefetcher_weight_bank_rm_axi_interconnect_axi_wlast),
    .axi_wvalid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid),
    .axi_wready                 (prefetcher_weight_bank_rm_axi_interconnect_axi_wready),
    .axi_bid                    (prefetcher_weight_bank_rm_axi_interconnect_axi_bid),
    .axi_bresp                  (prefetcher_weight_bank_rm_axi_interconnect_axi_bresp),
    .axi_bvalid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid),
    .axi_bready                 (prefetcher_weight_bank_rm_axi_interconnect_axi_bready),
    .axi_arid                   (prefetcher_weight_bank_rm_axi_interconnect_axi_arid),
    .axi_araddr                 (prefetcher_weight_bank_rm_axi_interconnect_axi_araddr),
    .axi_arlen                  (prefetcher_weight_bank_rm_axi_interconnect_axi_arlen),
    .axi_arsize                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arsize),
    .axi_arburst                (prefetcher_weight_bank_rm_axi_interconnect_axi_arburst),
    .axi_arlock                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arlock),
    .axi_arcache                (prefetcher_weight_bank_rm_axi_interconnect_axi_arcache),
    .axi_arprot                 (prefetcher_weight_bank_rm_axi_interconnect_axi_arprot),
    .axi_arqos                  (prefetcher_weight_bank_rm_axi_interconnect_axi_arqos),
    .axi_arregion               (), // not used
    .axi_arvalid                (prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid),
    .axi_arready                (prefetcher_weight_bank_rm_axi_interconnect_axi_arready),
    .axi_rid                    (prefetcher_weight_bank_rm_axi_interconnect_axi_rid),
    .axi_rdata                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rdata),
    .axi_rresp                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rresp),
    .axi_rlast                  (prefetcher_weight_bank_rm_axi_interconnect_axi_rlast),
    .axi_rvalid                 (prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid),
    .axi_rready                 (prefetcher_weight_bank_rm_axi_interconnect_axi_rready)
);

// ====================================================================================
// Prefetcher
// ====================================================================================

// prefetcher #(
//     .FETCH_TAG_COUNT (top_pkg::MESSAGE_CHANNEL_COUNT)
// ) prefetcher_i (
prefetcher prefetcher_i (
    .core_clk                                                  (clk),
    .resetn                                                    (!rst),

    // .regbank_clk                                               (regbank_clk),
    // .regbank_resetn                                            (regbank_resetn),

    // Node Scoreboard -> Prefetcher Interface
    .nsb_prefetcher_req_valid                                  (nsb_prefetcher_req_valid),
    .nsb_prefetcher_req_ready                                  (nsb_prefetcher_req_ready),
    .nsb_prefetcher_req                                        (nsb_prefetcher_req),
    .nsb_prefetcher_resp_valid                                 (nsb_prefetcher_resp_valid),
    .nsb_prefetcher_resp                                       (nsb_prefetcher_resp),

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



// ====================================================================================
// Interconnect
// ====================================================================================

// always_comb begin
//     prefetcher_weight_bank_rm_axi_interconnect_axi_araddr       = axi_araddr;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arburst       = axi_arburst;    
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arcache       = axi_arcache;      
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arid         = axi_arid;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arlen        = axi_arlen;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arlock       = axi_arlock;      
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arprot       = axi_arprot;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arqos        = axi_arqos;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arsize       = axi_arsize;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arvalid       = axi_arvalid;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_arready       = axi_arready;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awaddr       = axi_awaddr;  
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awburst       = axi_awburst;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awcache       = axi_awcache;      
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awid         = axi_awid;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awlen        = axi_awlen;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awlock       = axi_awlock;    
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awprot       = axi_awprot;      
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awqos        = axi_awqos;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awready       = axi_awready;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awsize       = axi_awsize;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_awvalid      = axi_awvalid;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_bid          = axi_bid;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_bready       = axi_bready;      
//     prefetcher_weight_bank_rm_axi_interconnect_axi_bresp        = axi_bresp;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_bvalid       = axi_bvalid;      
//     prefetcher_weight_bank_rm_axi_interconnect_axi_rdata        = axi_rdata;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_rid          = axi_rid;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_rlast        = axi_rlast;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_rready       = axi_rready;      
//     prefetcher_weight_bank_rm_axi_interconnect_axi_rresp        = axi_rresp;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_rvalid       = axi_rvalid;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_wdata        = axi_wdata;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_wlast        = axi_wlast;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_wready       = axi_wready;       
//     prefetcher_weight_bank_rm_axi_interconnect_axi_wstrb        = axi_wstrb;
//     prefetcher_weight_bank_rm_axi_interconnect_axi_wvalid       = axi_wvalid;      
// end


endmodule
