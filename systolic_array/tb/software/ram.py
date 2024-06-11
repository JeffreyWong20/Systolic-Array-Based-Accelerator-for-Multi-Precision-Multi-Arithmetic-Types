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
    generate the writeback offset with in different iterations to produce a section of result row with size of bus width
    """
    output_feature = weight_matrix_size[0]
    iteration_per_result_row = ceildiv(output_feature, systolic_array_size[1])
    for i in range(iteration_per_result_row):
        # if the offset is greater than 64 than we need to write the result in the next line
        yield (i * systolic_array_size[1] * byte_per_feature) // 64 ,(i * systolic_array_size[1] * byte_per_feature) % 64
        
     
def writeback_address_generator(writeback_address, input_matrix_size, weight_matrix_size, ultra_ram_size=1024, systolic_array_size=(4,128), byte_per_feature=4):
    """
    generate the start address for write back to the memory in different iterations and also offset
    input_matrix: input matrix
    """
    output_channel = input_matrix_size[0]
    output_feature = weight_matrix_size[0]
    block_channel_size = systolic_array_size[0]
    block_feature_size = systolic_array_size[1]
    
    byte_per_row = ceildiv(output_feature * byte_per_feature, 64) * 64 # 64 bytes alignment
    line_per_row = byte_per_row // 64               # how many line to store the result row
    line_per_sys_array = block_feature_size // 16   # how many line can be written by the systolic array in one iteration
    step = line_per_sys_array
    # can these line be produced by the systolic array in one iteration
    if line_per_sys_array > line_per_row:
        step = 1
        line_per_row = 1 
    else:
        # Only one beat is needed per iteration
        if line_per_sys_array == 0 or line_per_sys_array == 1:
            step = 1
            line_per_row = line_per_row
        else:
        # Multiple beats are needed per iteration 
            multi_beat = True
            print(f"line_per_sys_array={line_per_sys_array}, line_per_row={line_per_row}")
            print(f"output_feature={output_feature}, systolic_array_size[1]={systolic_array_size[1]}")
            print(f"output_channel={output_channel}, systolic_array_size[0]={systolic_array_size[0]}")
            # raise ValueError("Not supported yet.")
    
    byte_per_channel_block = block_channel_size * byte_per_row
    # NOTE: This does not work for all cases 
    # TODO: This might fail for some edge cases
    iteration_per_channel= ceildiv(output_channel, systolic_array_size[0])
    for i in range(iteration_per_channel):        
        for line, offset in offset_generator(weight_matrix_size, ultra_ram_size, systolic_array_size, byte_per_feature):
            if offset != 0 and multi_beat:
                print(f"Warning: unalignment acess with mutlibeat write is not supported yet.")
                raise ValueError("Not supported yet.")
            # NOTE:
            # Because of the offset machanism in the write back, we have to write feature with the lower memory index first (right most of the array)
            # A row can span multiple lines in the memory
            # Right most of the row is usually the bottom line of all thoses line. (Refer to the report for more detail)
            # TODO:
            # Obviously, this need to be change for more flexibility and edges cases. 
            current_line = line_per_row - step - line
            yield writeback_address + current_line * 64 + i * byte_per_channel_block, offset
        

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
    
def write_to_file(data, filename, start_address=0, each_feature_size=4, padding_alignment=64, writing_mode='w', direct_write_str=False):
    if not direct_write_str:
        data = np.asarray(data.detach().numpy(), dtype=np.int8)
    # if data is a 1D array, convert it to a 2D array
    num_features_in_a_row = data.shape[0] if len(data.shape) == 1 else data.shape[1]
    start_address_line = ceildiv(start_address, padding_alignment)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
            
    with open(filename, writing_mode) as f:
        print(f"Current number of lines: {len(lines)} and the specified line number: {start_address_line}")
        if start_address_line < 0 or start_address_line > len(lines):
            print("Warning: Line number out of range. Appending new lines up to the specified line number.")
            print(f"Current number of lines: {len(lines)} and the specified line number: {start_address_line}")
            while len(lines) < start_address_line:
                lines.append('\n')  # Append new lines until the desired line number is reached
                # write an empty line to the file which is 64 bytes aligned
                f.write('0'*64*2)
                f.write('\n')

        line = ''
        written_elements_in_a_wordline = 0
        for idx, val in enumerate(data.flatten()):
            if idx % num_features_in_a_row == 0 and idx != 0:
                line += '\n'
                f.writelines(line)
                written_elements_in_a_wordline = 0
                line = ''
            elif written_elements_in_a_wordline == 16:
                line += '\n'
                f.writelines(line)
                written_elements_in_a_wordline = 0
                line = ''
            # elif idx * each_feature_size % padding_alignment == 0 and idx != 0:
            #     line += '\n'
            #     f.writelines(line)
            #     written_elements_in_a_wordline = 0
            #     line = ''
            if not direct_write_str:
                line += "000000"
                line += struct.pack('b', val).hex()
            else:
                line += val
            written_elements_in_a_wordline +=1
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
    
async def read_ram(axi_ram_driver, software_result_matrix, byte_per_feature, result_start_address):
    software_result_matrix = software_result_matrix
    byte_per_row = ceildiv(software_result_matrix.shape[1]*byte_per_feature, 64) * 64
    line_per_row = byte_per_row // 64
    hardware_result_matrix = np.zeros(software_result_matrix.shape).astype(np.uint8)

    # Iterate through each row and line, process data, and populate the hardware result matrix
    for row in range(software_result_matrix.shape[0]):
        for i in range(line_per_row):
            data = await axi_ram_driver.axi_read(result_start_address + i*64 + row*byte_per_row)
            data_hex = data.hex()[2:] 
            # data hex should be 64 bytes so it should have 128 characters
            if len(data_hex) < 128:
                data_hex = '0'*int((128-len(data_hex))) + data_hex
            print(data_hex)
            
            # 1 line can at most contain 64//byte_per_feature = 16 features
            for element in range(64//byte_per_feature):
                idx = i*16 + element # per line is 16 features
                # if the feature in this transfer is less than 16, we need to skip the first few bytes
                if software_result_matrix.shape[1]-16*i < 16:
                    start_read_idx = (16-software_result_matrix.shape[1])*byte_per_feature*2
                else:
                    start_read_idx = 0
                
                if idx < software_result_matrix.shape[1]:
                    start_feature_idx = start_read_idx+element*byte_per_feature*2
                    end_feature_idx = start_read_idx+element*byte_per_feature*2+byte_per_feature*2
                    hardware_result_matrix[row][idx] = int(data_hex[start_feature_idx:end_feature_idx], 16)
                    # print(f"Error: row={row}, idx={idx}, start_feature_idx={start_feature_idx}, end_feature_idx={end_feature_idx}, data_hex={data_hex}")
                    # print(data_hex[start_feature_idx:end_feature_idx])
                    
    return hardware_result_matrix
    
if __name__ == "__main__":
    for i, (writeback_address, offset) in enumerate(writeback_address_generator(0x1200000, input_matrix_size=(4, 16), weight_matrix_size=(64, 16), systolic_array_size=(4,32))):
        print(f"writeback_address={hex(writeback_address)}, offset={offset}")

    print("")