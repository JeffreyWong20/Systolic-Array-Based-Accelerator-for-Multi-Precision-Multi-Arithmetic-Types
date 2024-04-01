# This file offers a simple test for an AXI ram module
import itertools
import logging
import os
import random
import struct
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

import cocotb
from cocotb.runner import get_runner
from cocotb.triggers import Timer, FallingEdge
from cocotbext.axi import AxiBus, AxiMaster, AxiRam

import torch
import torch.nn as nn
import logging
import numpy as np
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
    )
)
from systolic_array.tb.software.ram import RamTester, cycle_reset
from systolic_array.tb.software.linear import LinearInteger

np.random.seed(0)
debug = True
# create a logger client
logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)

# --------------------------------------------------
#   Model specifications
#   prefer small models for fast test
# --------------------------------------------------
config = {
    "weight_width": 8,
    "weight_frac_width": 0,
    "data_in_width": 8,
    "data_in_frac_width": 0,
    "bias_width": 8,
    "bias_frac_width": 0,
}
class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()
        input_features = 4
        output_features = 4
        random_matrix = np.random.randint(0, 101, size=(input_features, output_features))
        self.fc1 = LinearInteger(input_features, output_features, bias=False, config=config)
        self.fc1.weight.data = torch.tensor(random_matrix, dtype=torch.float32)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x

# --------------------------------------------------
#  Helper functions
# --------------------------------------------------
def debug_state(dut, state):
    logger.debug(
        """ (
            {} State: (
            nsb_prefetcher_req_valid,
            ) = ({}
            )""".format(
            state,
            dut.weight_prefetcher_req_valid.value,
        )
    )

def reset_nsb_prefetcher(dut):
    dut.weight_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.weight_prefetcher_req.out_features.value = 0                 # number of output features
    dut.weight_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests
    dut.feature_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.feature_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.feature_prefetcher_req.out_features.value = 0                 # number of output features
    dut.feature_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests

def reset_fte(dut):
    # input   logic                                                nsb_fte_req_valid,
    # output  logic                                                nsb_fte_req_ready,
    # input   NSB_FTE_REQ_t                                        nsb_fte_req,
    # output  logic                                                nsb_fte_resp_valid, // valid only for now
    # output  NSB_FTE_RESP_t                                       nsb_fte_resp,
    dut.nsb_fte_req_valid.value = 0
    dut.nsb_fte_req.precision.value = 0
    dut.nsb_fte_req.nodeslots.value = 0
    
def reset_all_axi_input_signals(dut):    
    dut.axi_awid.value = 0
    dut.axi_awaddr.value = 0
    dut.axi_awlen.value = 0
    dut.axi_awsize.value = 0
    dut.axi_awburst.value = 0
    dut.axi_awlock.value = 0
    dut.axi_awcache.value = 0
    dut.axi_awprot.value = 0
    dut.axi_awqos.value = 0
    dut.axi_awregion.value = 0
    dut.axi_awvalid.value = 0
    # dut.axi_awready.value = 0
    dut.axi_wdata.value = 0
    dut.axi_wstrb.value = 0
    dut.axi_wlast.value = 0
    dut.axi_wvalid.value = 0
    # dut.axi_wready.value = 0
    # dut.axi_bid.value = 0
    # dut.axi_bresp.value = 0
    # dut.axi_bvalid.value = 0
    dut.axi_bready.value = 0
    dut.axi_arid.value = 0
    dut.axi_araddr.value = 0
    dut.axi_arlen.value = 0
    dut.axi_arsize.value = 0
    dut.axi_arburst.value = 0
    dut.axi_arlock.value = 0
    dut.axi_arcache.value = 0
    dut.axi_arprot.value = 0
    dut.axi_arqos.value = 0
    dut.axi_arregion.value = 0
    dut.axi_arvalid.value = 0
    # dut.axi_arready.value = 0
    # dut.axi_rid.value = 0
    # dut.axi_rdata.value = 0
    # dut.axi_rresp.value = 0
    # dut.axi_rlast.value = 0
    # dut.axi_rvalid.value = 0
    dut.axi_rready.value = 0
    

# --------------------------------------------------
#  Actual test
# --------------------------------------------------
async def mlp_test(dut):
    
    mlp = MLP()
    # Input data
    input_data = torch.tensor(
        [[91, 30, 46, 20],
        [71, 57, 41, 71],
        [45, 42,  0, 12],
        [23, 47,  1, 31]], dtype=torch.float32)

    # Set weights for fc1 layer
    mlp.fc1.weight.data = torch.tensor(
        [[ 15, 107,  64,  46],
        [123, 116,   5,  85],
        [ 28,  12, 125,  88],
        [ 24,  75,  18,  29]], dtype=torch.float32)

    # Create a 10ns-period clock on port clk
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    await Timer(100, units="ns")
    clock = Clock(dut.clk, 10, units="ns")
    # Start the clock
    cocotb.start_soon(clock.start())
    # Reset cycle
    await Timer(20, units="ns")
    dut.rst.value = 1 
    reset_nsb_prefetcher(dut)
    reset_fte(dut)
    await Timer(100, units="ns")
    dut.rst.value = 0
    
    
    ram_tester = RamTester(dut.axi_ram)
    await ram_tester.initialize()
    # write weights to RAM
    weight = mlp.fc1.w_quantizer(mlp.fc1.weight)
    await ram_tester.write_to_ram(weight)
    await ram_tester.read_from_ram_and_verify(weight)
    # write input data to RAM
    weigth_address_range = mlp.fc1.weight.shape[0] * 16 * 4 # TODO: hardcode 16 and 4, assuming input channel is less than 16.
    await ram_tester.write_to_ram(input_data, start_address=weigth_address_range)
    await ram_tester.read_from_ram_and_verify(input_data, start_address=weigth_address_range)
    print("Done writing and finish verification")
    
    """
    # req_opcode =          dut.nsb_prefetcher_req.value[0:2]  
    # start_address =       dut.nsb_prefetcher_req.value[3:36]
    # in_features =         dut.nsb_prefetcher_req.value[37:47]
    # out_features =        dut.nsb_prefetcher_req.value[48:58]
    # nodeslot =            dut.nsb_prefetcher_req.value[59:63]
    # nodeslot_precision =  dut.nsb_prefetcher_req.value[64:65]
    # neighbour_count =     dut.nsb_prefetcher_req.value[66:75]
    """  
    # --------------------------------------------------
    dut.weight_prefetcher_req_valid.value = 1                               # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value   = 0                        # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value  = 0x0000                 # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value  = 4                        # number of input features
    dut.weight_prefetcher_req.out_features.value = 4                        # number of output features
    dut.weight_prefetcher_req.nodeslot.value     = 0                        # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = 1                  # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0                     # not used for weight bank requests
    # --------------------------------------------------
    dut.feature_prefetcher_req_valid.value = 1                              # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value   = 0                       # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = weigth_address_range  # start address of the weight bank
    dut.feature_prefetcher_req.in_features.value  = 4                       # number of input features
    dut.feature_prefetcher_req.out_features.value = 4                       # number of output features
    dut.feature_prefetcher_req.nodeslot.value     = 0                       # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = 1                 # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0                    # not used for weight bank requests
    # --------------------------------------------------
    dut.nsb_fte_req_valid.value = 1                                         # enable the fte
    dut.nsb_fte_req.precision.value = 1                                     # 01 is for fixed 8-bit precision
    # --------------------------------------------------
    print("Done instructing fte")
    
    i = 0
    while True:
        await FallingEdge(dut.clk)
        await Timer(10, units="ns")
        if dut.nsb_fte_resp_valid.value == 1:
            done = True
            break
        
        if i==1000:
            done = False
            break
        i+=1
        
    result_matrix = [[0]*4 for _ in range(4)]  # Initialize a 4x4 matrix with zeros
    for r in range(4):
        for c in range(4):
            print(ram_tester.axi_ram.read(0x200000000 + r*64 + c, 1).hex())
            result_matrix[r][c] = int(ram_tester.axi_ram.read(0x200000000 + r*64 + c, 1).hex(),16)
    print("Result matrix:", result_matrix)
    
    assert (
        done
    ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"
    

def test_axi_runner():
    """Simulate the adder example using the Python runner.
    This file can be run directly or via pytest discovery.
    """
    # sim = os.getenv("SIM", "verilator")
    # extra_args = [
    #     "--timescale",
    #     "1ps/1ps",
    #     "-Wno-WIDTH",
    #     "-Wno-CASEINCOMPLETE"
    # ]
    # wave_args = [
    #     "--trace-fst",
    #     "--trace-structs"
    # ]

    sim = "icarus"
    extra_args = []
    wave_args = []
    
    proj_path = Path(__file__).resolve().parent
    # equivalent to setting the PYTHONPATH environment variable

    verilog_sources = [
        proj_path / "include" / "top_pkg.sv",
        proj_path / "rtl" / "prefetcher.sv",
        proj_path / "rtl" / "prefetcher_weight_bank.sv",
        proj_path / "lib" / "buffer" / "ultraram.v",
        proj_path / "lib" / "buffer" / "ultraram_fifo.sv",
        proj_path / "lib" / "axi" / "axi_read_master.sv",
        proj_path / "lib" / "axi" / "axi_interface.sv",
        proj_path / "lib" / "axi" / "axi_ram.sv",
        proj_path / "top.sv",
    ]

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="top",
        always=True,
        build_args=extra_args+wave_args,
    )
    runner.test(hdl_toplevel="top", test_module="top_tb")


if __name__ == "__main__":
    # test_axi_runner()
    fc = MLP()
    
    input_data = torch.tensor(
        [[91, 30, 46, 20],
        [71, 57, 41, 71],
        [45, 42,  0, 12],
        [23, 47,  1, 31]], dtype=torch.float32)

    # Set weights for fc1 layer
    fc.fc1.weight.data = torch.tensor(
        [[ 15, 107,  64,  46],
        [123, 116,   5,  85],
        [ 28,  12, 125,  88],
        [ 24,  75,  18,  29]], dtype=torch.float32)

    # Quantize weights and input data
    quantized_weight = fc.fc1.w_quantizer(fc.fc1.weight.data)
    quantized_data = fc.fc1.w_quantizer(input_data)
    weight, data = quantized_weight.numpy().astype(np.int8), quantized_data.numpy().astype(np.int8)

    # Print quantized weights in hex format
    print("Quantized Weights:")
    for row in weight:
        hex_row = [hex(value) for value in row]
        print(hex_row)

    # Print quantized input data in hex format
    print("\nTransposed Quantized Input Data:")
    transposed_data = data.transpose()
    for row in transposed_data:
        hex_row = [hex(value) for value in row]
        print(hex_row)

    # Calculate and print linear operation result
    result = torch.nn.functional.linear(input_data, fc.fc1.weight.data).numpy().astype(np.int64)
    print("\nLinear Operation Result:")
    for row in result:
        hex_row = [hex(value)[-2:] for value in row]
        print(hex_row)
    for row in result:
        hex_row = [int(hex(value)[-2:],16) for value in row]
        print(hex_row)
        
    
    # Calculate the matrix multiplication result
    result = np.matmul(data, weight)
    print("\nMatrix Multiplication Result:")
    for row in result:
        hex_row = [hex(value)[-2:] for value in row]
        print(hex_row)