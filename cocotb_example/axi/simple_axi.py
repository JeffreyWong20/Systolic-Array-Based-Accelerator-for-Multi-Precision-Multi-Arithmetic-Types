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
from cocotb.triggers import Timer
from cocotbext.axi import AxiBus, AxiMaster, AxiRam
import torch
import numpy as np
from systolic_array.software.linear import LinearInteger

np.random.seed(0)
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
    dut.rst.setimmediatevalue(0)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

@cocotb.test()
async def simple_ram_test(dut):
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    axi_master = AxiMaster(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst)
    # each address is 1 byte
    axi_ram = AxiRam(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst, size=2**16)

    mlp = MLP()
    weight = mlp.fc1.w_quantizer(mlp.fc1.weight).detach().numpy()
    weight = weight.astype(np.int8)

    await cycle_reset(dut)

    for weight_idx, weight_val in enumerate(weight.flatten()):
        weight_bytes = struct.pack('b', weight_val)
        print("weight_id = {}, weight_val = {}, weight_bytes = x{} ".format(weight_idx, weight_val, weight_bytes.hex()))
        axi_ram.write(weight_idx, weight_bytes)

    for weight_idx, weight_val in enumerate(weight.flatten()):
        data = axi_ram.read(weight_idx, 1)
        weight_bytes = struct.pack('b', weight_val)
        assert (data == weight_bytes), "data = {}, data_bytes = x{}".format(data.hex(), weight_bytes.hex())

    # axi_ram.write(0x0000, b'test')

    # data = axi_ram.read(0x0000, 1)
    # data_1 = axi_ram.read(0x0001, 1)
    # print(data, data_1)
    # axi_ram.hexdump(0x0000, 8, prefix="RAM")
   
    # assert data == b'test'


def test_axi_runner():
    """Simulate the adder example using the Python runner.

    This file can be run directly or via pytest discovery.
    """
    sim = os.getenv("SIM", "verilator")

    proj_path = Path(__file__).resolve().parent.parent
    # equivalent to setting the PYTHONPATH environment variable

    verilog_sources = [proj_path / "axi" / "test_axi.sv"]
    print(verilog_sources)

    # equivalent to setting the PYTHONPATH environment variable
    sys.path.append(str(proj_path / "tests"))

    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="test_axi",
        always=True,
    )
    runner.test(hdl_toplevel="test_axi", test_module="simple_axi")


if __name__ == "__main__":
    test_axi_runner()