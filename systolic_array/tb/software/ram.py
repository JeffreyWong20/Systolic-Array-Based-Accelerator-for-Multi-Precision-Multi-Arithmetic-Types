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

ceildiv = lambda a, b: -(-a // b)

def partitioner(data, ultra_ram_size=1024, systolic_array_size=(4,128), systolic_forward=True):
    # We assume the matrix is not being transposed
    # This partitioner will partition the data into chunks that can be processed by the systolic array
    # Data block will then be written in the memory in a row_major fashion
    row_size = data.shape[1]
    num_iteration_per_row = ceildiv(row_size, ultra_ram_size)
    feeded_systolic_array_dim = systolic_array_size[0] if systolic_forward else systolic_array_size[1]
    col_size = data.shape[0]
    num_iteration_per_col = ceildiv(col_size, feeded_systolic_array_dim)

        
    print(f"num_iteration_per_row={num_iteration_per_row}, num_iteration_per_col={num_iteration_per_col}")
    
    for i in range(num_iteration_per_col):
        for j in range(num_iteration_per_row):
            yield data[i*feeded_systolic_array_dim:(i+1)*feeded_systolic_array_dim, j*ultra_ram_size:(j+1)*ultra_ram_size]            

def offset_generator(weight_matrix_size, ultra_ram_size, systolic_array_size, byte_per_feature=4):
    """
    generate the writebakc offset with in different iterations to produce the result row
    """
    output_feature = weight_matrix_size[0]
    iteration_per_result_row = ceildiv(output_feature, systolic_array_size[1])
    for i in range(iteration_per_result_row):
        yield i * systolic_array_size[1] * byte_per_feature
        
     
def writeback_address_generator(start_address, input_matrix_size, weight_matrix_size, ultra_ram_size=1024, systolic_array_size=(4,128), byte_per_feature=4):
    """
    generate the start address for write back to the memory in different iterations and also offset
    input_matrix: input matrix
    """
    output_channel = input_matrix_size[0]
    output_feature = weight_matrix_size[0]
    block_channel_size = systolic_array_size[0]
    block_feature_size = systolic_array_size[1]
    
    byte_per_row = ceildiv(output_feature * byte_per_feature, 64) * 64 # 64 bytes alignment
    byte_per_channel_block = block_channel_size * byte_per_row
    
    iteration_per_channel= ceildiv(output_channel, systolic_array_size[0])
    for i in range(iteration_per_channel):
        for offset in offset_generator(weight_matrix_size, ultra_ram_size, systolic_array_size, byte_per_feature):
            yield start_address + i * byte_per_channel_block, offset
        

def load_address_generator(start_address, input_matrix_size, weight_matrix_size, ultra_ram_size=1024, systolic_array_size=(4,128), byte_per_feature=4):
    """
    generate the start address for loading the input matrix in different iterations
    input_matrix: input matrix
    """
    input_channel = input_matrix_size[0]
    input_feature = input_matrix_size[1]
    block_channel_size = systolic_array_size[0]
    block_feature_size = systolic_array_size[1]
    
    byte_per_row = ceildiv(input_feature * byte_per_feature, 64) * 64 # 64 bytes alignment
    byte_per_channel_block = block_channel_size * byte_per_row
    
    iteration_per_channel= ceildiv(input_channel, systolic_array_size[0])
    for i in range(iteration_per_channel):
        yield start_address + i * byte_per_channel_block
    
    
    
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
    
def write_to_file(data, filename, start_address=0, each_feature_size=4, padding_alignment=64, writing_mode='w'):
    data = np.asarray(data.detach().numpy(), dtype=np.int8)
    num_features_in_a_row = data.shape[1]
    start_address_line = ceildiv(start_address, padding_alignment) + 1
    
    with open(filename, 'r') as f:
        lines = f.readlines()

        if start_address_line < 0 or start_address_line > len(lines):
            print("Warning: Line number out of range. Appending new lines up to the specified line number.")
            while len(lines) < start_address_line:
                lines.append('\n')  # Append new lines until the desired line number is reached
            
            
    with open(filename, writing_mode) as f:
        while len(lines) < start_address_line:
            f.write("\n") 

        line = ''
        for idx, val in enumerate(data.flatten()):
            if idx % num_features_in_a_row == 0 and idx != 0:
                line += '\n'
                f.writelines(line)
                line = ''
            elif idx * each_feature_size % padding_alignment == 0 and idx != 0:
                line += '\n'
                f.writelines(line)
                line = ''
            
            line += "000000"
            line += struct.pack('b', val).hex()
            # print(f"Writing: index={idx}, value={val}, bytes=x{struct.pack('b', val).hex()}")
        f.writelines(line)
        f.write('\n')

class RamTester:
    """This is a ram class that also contains helper function to write and verify its functionality"""
    def __init__(self, dut, padding_aligment=64, each_feature_size=4):
        """_summary_
        :param max_in_features: max number of feature stored in a row of the feature bank
        :param each_feature_size: size of each feature in bytes
        """
        self.dut = dut
        self.axi_ram = AxiRam(AxiBus.from_prefix(dut, "axi"), dut.clk, dut.rst, size=2**34)
        self.each_feature_size = each_feature_size # each feature is 4 bytes
        self.padding_aligment = padding_aligment # 64 bytes alignment
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
        num_features_in_a_row = data.shape[1]
        self.each_row_size = ceildiv(num_features_in_a_row * self.each_feature_size, self.padding_aligment) * self.padding_aligment
        
        for idx, val in enumerate(data.flatten()):
            row_idx = (idx // num_features_in_a_row) * self.each_row_size
            feature_idx = (idx % num_features_in_a_row) * self.each_feature_size
            idx = row_idx + feature_idx + start_address
            
            byte_val = struct.pack('b', val)
            print(f"Writing: index={idx}, value={val}, bytes=x{byte_val.hex()}")
            self.axi_ram.write(idx, byte_val)
        
        
    async def read_from_ram_and_verify(self, data, start_address=0):
        data = np.asarray(data.detach().numpy(), dtype=np.int8)
        num_features_in_a_row = data.shape[1]
        self.each_row_size = ceildiv(num_features_in_a_row * self.each_feature_size, self.padding_aligment) * self.padding_aligment
        
        for idx, val in enumerate(data.flatten()):
            row_idx = (idx // num_features_in_a_row) * self.each_row_size
            feature_idx = (idx % num_features_in_a_row) * self.each_feature_size
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