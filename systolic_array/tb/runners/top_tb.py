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

async def cycle_reset(dut):
    """Reset the dut for one clock cycle"""
    dut.rst.setimmediatevalue(0)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


class RamTester:
    """Helper class to test the RAM module"""
    def __init__(self, dut):
        self.dut = dut
        # self.axi_master = AxiMaster(AxiBus.from_prefix(dut, "s_axi"), dut.clk, dut.rst)
        self.axi_ram = AxiRam(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst, size=2**16)
        self.mlp = MLP()

    async def initialize(self):
        await cycle_reset(self.dut)

    async def write_to_ram(self, data):
        data = data.detach().numpy().astype(np.int8)
        for idx, val in enumerate(data.flatten()):
            byte_val = struct.pack('b', val)
            print(f"Writing: index={idx}, value={val}, bytes=x{byte_val.hex()}")
            # await self.axi_master.write(idx, byte_val)
            self.axi_ram.write(idx, byte_val)
            print("Done writing")

    async def read_from_ram_and_verify(self, data):
        data = data.detach().numpy().astype(np.int8)
        for idx, val in enumerate(data.flatten()):
            expected_bytes = struct.pack('b', val)
            print(f"index={idx}, value={val}, bytes=x{expected_bytes.hex()}")
            # read_data = await self.axi_master.read(idx, 1)
            read_data = self.axi_ram.read(idx, 1)
            print("Done reading")
            # print(f"Reading: index={idx}, value={val}, bytes=x{read_data.data.hex}")
            print(f"Reading: index={idx}, value={val}, bytes=x{read_data}")
            # assert read_data.data == expected_bytes, f"Data mismatch at index {idx}: " \
            #                                      f"read={read_data.data.hex()}, expected={expected_bytes.hex()}"
            assert read_data == expected_bytes, f"Data mismatch at index {idx}: " \
                                                 f"read={read_data.hex()}, expected={expected_bytes.hex()}"

async def test_ram_operations(dut):
    ram_tester = RamTester(dut.axi_ram)
    await ram_tester.initialize()

    # Generate data or retrieve it from somewhere (replace this line with your actual data)
    data = ram_tester.mlp.fc1.w_quantizer(ram_tester.mlp.fc1.weight)
    
    await ram_tester.write_to_ram(data)
    await ram_tester.read_from_ram_and_verify(data)
    print("Done complete")
    # await ram_tester.read_from_ram_and_verify(data)


# @cocotb.test()
async def simple_ram_test(dut):
    dut = dut.axi_ram
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    
    axi_master = AxiMaster(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst)
    axi_ram = AxiRam(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst, size=2**16)

    await cycle_reset(dut)
    
    print("Writing to address 0x0000")
    # await axi_master.write(0x0000, b'test')
    print("Reading from address 0x0000")
    data = await axi_master.read(0x0000, 4)
    print("Data read: ", data)
    # axi_master.init_write(0x0000, b'test')
    axi_ram.write(0x0000, b'test')
    data = axi_ram.read(0x0000, 4)
    axi_ram.hexdump(0x0000, 8, prefix="RAM")
    
    assert data == b'test'


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
            dut.nsb_prefetcher_req_valid.value,
        )
    )

def reset_nsb_prefetcher(dut):
    dut.nsb_prefetcher_req_valid.value = 0                  # enable the prefetcher
    dut.nsb_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.nsb_prefetcher_req.start_address.value  = 0x0000            # start address of the weight bank
    dut.nsb_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.nsb_prefetcher_req.out_features.value = 0                 # number of output features
    dut.nsb_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.nsb_prefetcher_req.nodeslot_precision.value = 0                 # 01 is for fixed 8-bit precision
    dut.nsb_prefetcher_req.neighbour_count.value = 0                 # not used for weight bank requests

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
    
# @cocotb.test()
async def mlp_test(dut):
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
    await Timer(100, units="ns")
    dut.rst.value = 0
    
    
    # This code does not work
    ram_tester = RamTester(dut.axi_ram)
    await ram_tester.initialize()
    data = ram_tester.mlp.fc1.w_quantizer(ram_tester.mlp.fc1.weight)
    data = torch.cat((data, data), dim=0)
    await ram_tester.write_to_ram(data)
    await ram_tester.read_from_ram_and_verify(data)
    print("Done writing and reading")
    
    """
    typedef struct packed {
        NSB_PREF_OPCODE_e                      req_opcode;          // 3 bits
        logic [AXI_ADDRESS_WIDTH-1:0]          start_address;       // 34 bits

        // Weight bank requests 
        logic [$clog2(MAX_FEATURE_COUNT):0]    in_features;         // 11 bits
        logic [$clog2(MAX_FEATURE_COUNT):0]    out_features;        // 11 bits

        // For feature bank requests
        logic [$clog2(MAX_NODESLOT_COUNT)-1:0] nodeslot;            // 5 bits
        NODE_PRECISION_e                       nodeslot_precision;  // 2 bits
        logic [$clog2(MAX_NEIGHBOURS)-1:0]     neighbour_count;     // 10 bits
    } NSB_PREF_REQ_t;

    # req_opcode =          dut.nsb_prefetcher_req.value[0:2]  
    # start_address =       dut.nsb_prefetcher_req.value[3:36]
    # in_features =         dut.nsb_prefetcher_req.value[37:47]
    # out_features =        dut.nsb_prefetcher_req.value[48:58]
    # nodeslot =            dut.nsb_prefetcher_req.value[59:63]
    # nodeslot_precision =  dut.nsb_prefetcher_req.value[64:65]
    # neighbour_count =     dut.nsb_prefetcher_req.value[66:75]
    """
    # await cycle_reset(dut)

    # dut.nsb_prefetcher_req_valid.value = 1                  # enable the prefetcher
    # dut.nsb_prefetcher_req.value[0:2]   = 0                 # 00 is for weight bank requests
    # dut.nsb_prefetcher_req.value[3:36]  = 0x0000            # start address of the weight bank
    # dut.nsb_prefetcher_req.value[37:47] = 4                 # number of input features
    # dut.nsb_prefetcher_req.value[48:58] = 4                 # number of output features
    # dut.nsb_prefetcher_req.value[59:63] = 0                 # not used for weight bank requests
    # dut.nsb_prefetcher_req.value[64:65] = 1                 # 01 is for fixed 8-bit precision
    # dut.nsb_prefetcher_req.value[66:75] = 0                 # not used for weight bank requests
    # print("Done writing")
    # reset_all_axi_input_signals(dut.axi_ram)
    
    dut.nsb_prefetcher_req_valid.value = 1                      # enable the prefetcher
    dut.nsb_prefetcher_req.req_opcode.value   = 0               # 00 is for weight bank requests
    dut.nsb_prefetcher_req.start_address.value  = 0x0000        # start address of the weight bank
    dut.nsb_prefetcher_req.in_features.value  = 4               # number of input features
    dut.nsb_prefetcher_req.out_features.value = 4               # number of output features
    dut.nsb_prefetcher_req.nodeslot.value     = 0               # not used for weight bank requests
    dut.nsb_prefetcher_req.nodeslot_precision.value = 1         # 01 is for fixed 8-bit precision
    dut.nsb_prefetcher_req.neighbour_count.value = 0            # not used for weight bank requests
    print("Done writing")
    
    # Create a binary string with the specified bit values
    # binary_value = (
    #     '000'                       # 0 is for weight bank requests
    #     + '0000'                    # Start address of the weight bank
    #     + bin(4)[2:].zfill(11)      # Number of input features      remove the 0b prefix and pad with 0s to 11 bits
    #     + bin(4)[2:].zfill(11)      # Number of output features
    #     + '0' * 5                   # Not used for weight bank requests
    #     + '01'                      # 01 is for fixed 8-bit precision
    #     + '0' * 10                  # Not used for weight bank requests
    # )
    # # Convert the binary string to an integer
    # dut.nsb_prefetcher_req.value = int(binary_value, 2)
    # debug_state(dut, "Pre-clk")
    await Timer(1000, units="ns")
    # print("first clock")
    # # debug_state(dut, "Post-clk")
    # # debug_state(dut, "Pre-clk")
    # await FallingEdge(dut.clk)
    # print("first clock")
    # debug_state(dut, "Post-clk")

    # # done = False
    # # # Set a timeout to avoid deadlock
    # for i in range(100):
    #     print("first clock", i)
    #     await RisingEdge(dut.clk)
    #     await Timer(1, units="ns")
    #     await Timer(1, units="ns")
        
    #     # debug_state(dut, "Post-clk")
    #     # dut.weight_valid.value = test_case.weight.pre_compute()
    #     # dut.bias_valid.value = test_case.bias.pre_compute()
    #     # dut.data_in_valid.value = test_case.data_in.pre_compute()
    #     # await Timer(1, units="ns")
    #     # dut.data_out_ready.value = test_case.outputs.pre_compute(
    #     #     dut.data_out_valid.value
    #     # )
    #     # await Timer(1, units="ns")
    #     # debug_state(dut, "Post-clk")

    #     # dut.bias_valid.value, dut.bias.value = test_case.bias.compute(
    #     #     dut.bias_ready.value
    #     # )
    #     # dut.weight_valid.value, dut.weight.value = test_case.weight.compute(
    #     #     dut.weight_ready.value
    #     # )
    #     # dut.data_in_valid.value, dut.data_in.value = test_case.data_in.compute(
    #     #     dut.data_in_ready.value
    #     # )
    #     # await Timer(1, units="ns")
    #     # dut.data_out_ready.value = test_case.outputs.compute(
    #     #     dut.data_out_valid.value, dut.data_out.value
    #     # )
    #     debug_state(dut, "Pre-clk")
    #     if (
    #         i==19
    #     ):
    #         done = True
    #         break
    # completed = True
    # assert (
    #     done
    # ), "Deadlock detected or the simulation reaches the maximum cycle limit (fixed it by adjusting the loop trip count)"



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
    # extra_args = [
    #     "--timescale",
    #     "1ps/1ps",
    # ]

    # wave_args = [
    #     "WAVE=1"
    # ]

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
    test_axi_runner()
    # test_ram_operations()
    MLP = MLP()