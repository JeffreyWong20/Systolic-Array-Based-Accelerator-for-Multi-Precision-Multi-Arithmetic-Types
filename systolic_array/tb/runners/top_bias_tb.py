# This file offers a simple test for an AXI ram module
import logging
import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ReadOnly
from cocotb.runner import get_runner

import torch
import torch.nn as nn
import logging
import numpy as np
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        ".."
    )
)
from systolic_array.tb.software.axi_driver import AXIDriver
from systolic_array.tb.software.instruction import calculate_linear_and_writeback_b, load_feature_block_instruction_b, load_weight_block_instruction_b, clear_all
from systolic_array.tb.software.ram import read_ram, write_to_file, writeback_address_generator
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
        self.fc1 = LinearInteger(input_features, output_features, bias=True, config=config)
        self.fc1.weight.data = torch.tensor(random_matrix, dtype=torch.float32)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x

# --------------------------------------------------
#  Software groundtruth
# --------------------------------------------------
def compute_quantize(input_data, weight_data, bias_data, debug=False):
    """
    Compute the quantized result of the matrix multiplication between the input data and the weight data
    """ 
    mlp = MLP()
    mlp.fc1.weight.data = weight_data
    mlp.fc1.bias.data = bias_data
    quantized_weight = mlp.fc1.w_quantizer(mlp.fc1.weight.data)
    quantized_data = mlp.fc1.w_quantizer(input_data)
    quantized_bias = mlp.fc1.w_quantizer(mlp.fc1.bias.data)
    
    weight, data, bias = quantized_weight.numpy().astype(np.uint8), quantized_data.numpy().astype(np.uint8), quantized_bias.numpy().astype(np.uint8)
    result = torch.nn.functional.linear(input_data, mlp.fc1.weight.data,  mlp.fc1.bias.data).numpy().astype(np.int64)
    # cast the result to int8
    int_8_result = result.astype(np.uint8)
    if debug:
        print("Quantized Weights:")
        for row in weight:
            hex_row = [hex(value) for value in row]
            print(hex_row)
        print("Quantized Bias:")
        hex_row = [hex(value) for value in bias]
        print(hex_row)
        print("Quantized Input:")
        for row in data:
            hex_row = [hex(value) for value in row]
            print(hex_row)
        print("\nTransposed Quantized Weights:")
        transposed_weight = mlp.fc1.weight.data.transpose(0, 1).numpy().astype(np.uint8)
        for row in transposed_weight:
            hex_row = [hex(value) for value in row]
            print(hex_row)
        print("\nQuantized Linear Operation Result:")
        for row in int_8_result:
            hex_row = [hex(value) for value in row]
            print(hex_row)
        print("\nMaxtrix Multiplication without bias")
        result_mult = (input_data @ transposed_weight).numpy().astype(np.uint8)
        for row in result_mult:
            hex_row = [hex(value) for value in row]
            print(hex_row)
        print("\nMaxtrix Multiplication Result with transposed_weight with bias:")
        for row in result_mult:
            row+= mlp.fc1.bias.data.numpy().astype(np.uint8)
            hex_row = [hex(value) for value in row]
            print(hex_row)
    return data, weight, int_8_result

ceildiv = lambda a, b: -(-a // b)

# --------------------------------------------------
#  Actual test ::: MAKE SURE TO RUN THIS SCRIPT 
# --------------------------------------------------
input_matrix_size = (1, 32)
weight_matrix_size = (32, 32)
systolic_array_size = (1*4, 4*4)
byte_per_feature = 4
precision = 8
torch.manual_seed(42)  
mlp = MLP()
input_data = torch.randint(-128, 127, size=input_matrix_size, dtype=torch.float32)
mlp.fc1.weight.data = torch.randint(-128, 127, size=weight_matrix_size, dtype=torch.float32)
mlp.fc1.bias.data = torch.randint(-128, 127, size=(weight_matrix_size[0],), dtype=torch.float32)
result_start_address = 0x2000000
weight_start_address = 0x0000000

async def bias_test(dut):
    weigth_address_range = ceildiv(mlp.fc1.weight.shape[1] * 4, 64) * 64 * mlp.fc1.weight.shape[0] # A single row has to be multiple of 64 bytes and hence the start address has to be aligned to 64 bytes

    # Create a 10ns-period clock on port clk and reset port rst
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    #reset everything
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 1 
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await clear_all(dut)
    logger.info("Done clearing and reset")
    
    # write weights to RAM
    axi_ram_driver = AXIDriver(dut.ram_model)
    write_to_file(mlp.fc1.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='w',start_address=0, each_feature_size=4, padding_alignment=64)
    write_to_file(input_data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weigth_address_range, each_feature_size=4, padding_alignment=64)
        
    # Instruct the core
    byte_per_weight_block = ceildiv(mlp.fc1.weight.shape[1] * 4, 64) * 64 * systolic_array_size[1]
    byte_per_input_block = ceildiv(input_matrix_size[1] * 4, 64) * 64 * systolic_array_size[0]
    input_matrix_iteration = ceildiv(input_matrix_size[0], systolic_array_size[0])  # number channel blocks
    weight_matrix_iteration = ceildiv(weight_matrix_size[0], systolic_array_size[1]) # number of iteration to produce one output channel blocks
    timeout = max(weight_matrix_size[0] * weight_matrix_size[1], input_matrix_size[0], input_matrix_size[1]) * 100

    logger.info("Start executing instructions")
    for i, (writeback_address, offset) in enumerate(writeback_address_generator(writeback_address=result_start_address, input_matrix_size=input_matrix_size, weight_matrix_size=weight_matrix_size, systolic_array_size=systolic_array_size)):
        reversed_index =  (weight_matrix_iteration-1-(i % weight_matrix_iteration))
        weight_start_address = byte_per_weight_block * reversed_index
        input_start_address = weigth_address_range + byte_per_input_block * (i // weight_matrix_iteration)
        logger.info(f"Start address: {hex(writeback_address)}, Offset: {offset}, Load weight from: {hex(weight_start_address)}, Load input from: {hex(input_start_address)}")

        await RisingEdge(dut.clk)
        await load_weight_block_instruction_b(dut, start_address=weight_start_address, weight_block_size=weight_matrix_size, timeout=timeout)
        logger.info("Done instructing weight prefetcher")

        await RisingEdge(dut.clk)
        await load_feature_block_instruction_b(dut, start_address=input_start_address, input_block_size=input_matrix_size, timeout=timeout)
        logger.info("Done instructing feature prefetcher")

        await RisingEdge(dut.clk)
        bias = mlp.fc1.bias.data.numpy().astype(np.uint8).tolist()
        bias_section = bias[reversed_index*systolic_array_size[1]:(reversed_index+1)*systolic_array_size[1]]
        logger.info("Feeding bias:")
        logger.info(bias_section)
        await calculate_linear_and_writeback_b(dut, writeback_address=writeback_address, offset=offset, output_matrix_size=(input_matrix_size[0], weight_matrix_size[0]), bias=bias_section, timeout=timeout)
        logger.info("Done instructing fte")

        await RisingEdge(dut.clk)
    

    # Read the result from the RAM and compare it with the software result
    software_result_matrix = compute_quantize(input_data, mlp.fc1.weight.data, mlp.fc1.bias.data, debug=True)[2]
    hardware_result_matrix = await read_ram(axi_ram_driver, software_result_matrix, byte_per_feature, result_start_address)

    # Logging results
    logger.info("Hardware matrix:")
    logger.info(hardware_result_matrix)

    logger.info("Software matrix:")
    logger.info(software_result_matrix)

    # Assert the equality of hardware and software matrices
    logger.info("Comparing hardware and software matrices...")
    assert ((hardware_result_matrix == software_result_matrix).all()), "The solution is not correct"
    logger.info("Matrices match: Hardware and software solutions are equivalent.")
        

async def run_test(dut):
    logger.debug("running test inside run_test")
    input_matrix_size = (8, 128)
    weight_matrix_size = (4, 128)
    
    mlp = MLP()
    torch.manual_seed(42)
    input_data = torch.randint(0, 128, size=input_matrix_size, dtype=torch.float32)
    mlp.fc1.weight.data = torch.randint(0, 128, size=weight_matrix_size, dtype=torch.float32)
    mlp.fc1.bias.data = torch.randint(0, 128, size=(weight_matrix_size[0],), dtype=torch.float32)

    # Create a 10ns-period clock on port clk and reset port rst
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    #reset everything
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 1 
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    axi_ram_driver = AXIDriver(dut.ram_model)
    # await axi_ram_driver.axi_write(0x0, 0xaabbccdd）
    weigth_address_range = ceildiv(mlp.fc1.weight.shape[1] * 4, 64) * 64 * mlp.fc1.weight.shape[0] # A single row has to be multiple of 64 bytes and hence the start address has to be aligned to 64 bytes

    write_to_file(mlp.fc1.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", start_address=0, each_feature_size=4, padding_alignment=64)
    write_to_file(input_data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", start_address=weigth_address_range, each_feature_size=4, padding_alignment=64)
    data = await axi_ram_driver.axi_read(0x0)
    data = data.hex()[2:] # remove the 0x
    # await axi_ram_driver.axi_read(0x1)
    
    
def test_axi_runner():
    """
    Simulate the adder example using the Python runner.
    This file can be run directly or via pytest discovery.
    NOTE: This is not needed anymore, as we are primarily using modelsim
    """
    sim = "icarus"
    extra_args = [] #  "--timescale", "1ps/1ps", "-Wno-WIDTH", "-Wno-CASEINCOMPLETE"
    wave_args = []  #  "--trace-fst", "--trace-structs"
    
    # equivalent to setting the PYTHONPATH environment variable
    proj_path = Path(__file__).resolve().parent
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
    weight = mlp.fc1.w_quantizer(mlp.fc1.weight)
    bias = mlp.fc1.w_quantizer(mlp.fc1.bias)
    weigth_address_range = ceildiv(mlp.fc1.weight.shape[1] * 4, 64) * 64 * mlp.fc1.weight.shape[0] # A single row has to be multiple of 64 bytes and hence the start address has to be aligned to 64 bytes
    # The memory session after the weight matrix is dedicated to the bias matrix
    bias_address_range = ceildiv(mlp.fc1.bias.shape[0] * 4, 64) * 64 * 1 + weigth_address_range
    
    
    quantized_weight = mlp.fc1.w_quantizer(mlp.fc1.weight.data)
    quantized_data = mlp.fc1.w_quantizer(input_data)
    quantized_bias = mlp.fc1.w_quantizer(mlp.fc1.bias.data)
    
    weight, data, bias = quantized_weight.numpy().astype(np.uint8), quantized_data.numpy().astype(np.uint8), quantized_bias.numpy().astype(np.uint8)
    result = torch.nn.functional.linear(input_data, mlp.fc1.weight.data, mlp.fc1.bias.data).numpy().astype(np.int64)
    int_8_result = result.astype(np.uint8)
    
    print("Quantized Weights:")
    for row in weight:
        hex_row = [hex(value) for value in row]
        print(hex_row)
    print("Quantized Bias:")
    hex_row = [hex(value) for value in bias]
    print(hex_row)
    print("Quantized Input:")
    for row in data:
        hex_row = [hex(value) for value in row]
        print(hex_row)
    print("\nTransposed Quantized Weights:")
    transposed_weight = mlp.fc1.weight.data.transpose(0, 1).numpy().astype(np.uint8)
    for row in transposed_weight:
        hex_row = [hex(value) for value in row]
        print(hex_row)
    print("\nQuantized Linear Operation Result:")
    for row in int_8_result:
        hex_row = [hex(value) for value in row]
        print(hex_row)
    print("\nMaxtrix Multiplication Result with transposed_weight:")
    result_mult = (input_data @ transposed_weight).numpy().astype(np.uint8)
    for row in result_mult:
        row += mlp.fc1.bias.data.numpy().astype(np.uint8)
        hex_row = [hex(value) for value in row]
        print(hex_row)
    
    open("/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", 'w').close()
    write_to_file(mlp.fc1.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='w',start_address=0, each_feature_size=4, padding_alignment=64)
    # write_to_file(mlp.fc1.bias.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a',start_address=weigth_address_range, each_feature_size=4, padding_alignment=64)
    write_to_file(input_data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weigth_address_range, each_feature_size=4, padding_alignment=64)