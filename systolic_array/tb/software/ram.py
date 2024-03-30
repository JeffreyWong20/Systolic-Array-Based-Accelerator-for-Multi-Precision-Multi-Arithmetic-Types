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
    """This is a ram class that also contains helper function to write and verify its functionality"""
    def __init__(self, dut, max_in_features=16, each_feature_size=4):
        """_summary_
        :param max_in_features: max number of feature stored in a row of the feature bank
        :param each_feature_size: size of each feature in bytes
        """
        self.dut = dut
        self.axi_ram = AxiRam(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst, size=2**34)
        self.max_in_features = max_in_features # each row can store 16 features
        self.each_feature_size = each_feature_size # each feature is 4 bytes
        self.each_row_size = self.max_in_features * self.each_feature_size
        # --------------------------------------------------
        # |         |           |           |           | ... in total 16 features
        # |f1*0*0*0 |   f2      |   f3      |   f4      | ... in total 16 features
        # |         |           |           |           | ... in total 16 features
        # |         |           |           |           | ... in total 16 features
        # --------------------------------------------------

    async def initialize(self):
        await cycle_reset(self.dut)
        
        
    async def write_to_ram(self, data, start_address=0):
        data = np.asarray(data.detach().numpy(), dtype=np.int8)
        num_features = data.shape[0]
        
        for idx, val in enumerate(data.flatten()):
            row_idx = (idx // num_features) * self.each_row_size
            feature_idx = (idx % num_features) * self.each_feature_size
            idx = row_idx + feature_idx + start_address
            
            byte_val = struct.pack('b', val)
            print(f"Writing: index={idx}, value={val}, bytes=x{byte_val.hex()}")
            self.axi_ram.write(idx, byte_val)
        
        
    async def read_from_ram_and_verify(self, data, start_address=0):
        data = np.asarray(data.detach().numpy(), dtype=np.int8)
        num_features = data.shape[0]
        
        for idx, val in enumerate(data.flatten()):
            row_idx = (idx // num_features) * self.each_row_size
            feature_idx = (idx % num_features) * self.each_feature_size
            idx = row_idx + feature_idx + start_address
            
            expected_bytes = struct.pack('b', val)            
            read_data = self.axi_ram.read(idx, 1)
            print(f"Reading: index={idx}, value={val}, bytes=x{read_data}")
            
            assert read_data == expected_bytes, \
                f"Data mismatch at index {idx}: read={read_data.hex()}, expected={expected_bytes.hex()}"


async def test_ram_operations(dut):
    ram_tester = RamTester(dut.axi_ram)
    await ram_tester.initialize()
    
    mlp = MLP()
    data = mlp.fc1.w_quantizer(ram_tester.mlp.fc1.weight)
    
    await ram_tester.write_to_ram(data)
    await ram_tester.read_from_ram_and_verify(data)
    print("Done complete")


# @cocotb.test()
async def simple_ram_test(dut):
    dut = dut.axi_ram
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())
    
    axi_master = AxiMaster(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst)
    axi_ram = AxiRam(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst, size=2**16)
    await cycle_reset(dut)
    
    await axi_master.write(0x0000, b'test') # write 4 bytes to address 0x0000
    data = await axi_master.read(0x0000, 4) # read 4 bytes from address 0x0000
    print("Data read: ", data)
    # axi_master.init_write(0x0000, b'test')
    axi_ram.write(0x0000, b'test')
    data = axi_ram.read(0x0000, 4)
    axi_ram.hexdump(0x0000, 8, prefix="RAM")
    
    assert data == b'test'
    
if __name__ == "__main__":
    test_ram_operations()