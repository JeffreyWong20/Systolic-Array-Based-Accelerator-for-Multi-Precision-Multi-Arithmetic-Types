# This file offers a testbench for the mixed precision layer.
import math
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
from systolic_array.tb.software.mixed_precision import integer_quantize_in_hex
from systolic_array.tb.software.axi_driver import AXIDriver
from systolic_array.tb.software.instruction import calculate_linear_and_writeback_b, load_feature_block_instruction_b, load_weight_block_instruction_b, clear_all, Activation
from systolic_array.tb.software.ram import read_ram, write_to_file, writeback_address_generator
from systolic_array.tb.software.linear import LinearInteger, LinearMixedPrecision

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
SYSTOLIC_MODULE_COUNT = 4
systolic_array_size = (1*4, SYSTOLIC_MODULE_COUNT*4)
byte_per_feature, precision = 4, 8

input_shape = (1, 64) # NOTE: IF you change this, you need to run this file to generate the weight and bias
first_layer_weight_shape = (32, 64) # NOTE: IF you change this, you need to run this file to generate the weight and bias

input_start_address = 0x0010000
weight_start_address = 0x0000000
first_layer_result = 0x1200000

high_precision = (8, 4)
low_precision = (4, 3)

config = {
    # High precision
    "weight_high_width": high_precision[0],
    "weight_high_frac_width": high_precision[1],
    "data_in_high_width": high_precision[0],
    "data_in_high_frac_width": high_precision[1],
    "bias_high_width": high_precision[0],
    "bias_high_frac_width": high_precision[1],
    # Low precision
    "weight_low_width": low_precision[0],
    "weight_low_frac_width": low_precision[1],
    "data_in_low_width": low_precision[0],
    "data_in_low_frac_width": low_precision[1],
    "bias_low_width": low_precision[0],
    "bias_low_frac_width": low_precision[1],
    #
    "block_width": int(systolic_array_size[1]),
    "block_high_width": int(systolic_array_size[1]/2),
    "block_height": 4,
    # 
    "weight_width": 8,
    "weight_frac_width": 4,
    "data_IN_WIDTH": 8,
    "data_IN_FRAC_WIDTH": 4,
    "bias_width": 8,
    "bias_frac_width": 4,
    "bypass": False,
}
class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self, first_layer_weight_shape) -> None:
        super().__init__()
        self.fc1 = LinearMixedPrecision(first_layer_weight_shape[1], first_layer_weight_shape[0], bias=True, config=config)
        self.fc_no_bias = LinearMixedPrecision(first_layer_weight_shape[1], first_layer_weight_shape[0], bias=False, config=config)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        # x = torch.nn.functional.relu(self.fc1(x))
        return x

ceildiv = lambda a, b: -(-a // b)
# --------------------------------------------------
#  Global setting
# --------------------------------------------------
torch.manual_seed(42)  
feature_start_address_list = [input_start_address, first_layer_result]
input_data = torch.randn(*(input_shape))

fc = MLP(first_layer_weight_shape)
fc.fc1.weight.data = torch.randn(*first_layer_weight_shape) 
fc.fc1.bias.data = torch.randn(first_layer_weight_shape[0])
fc.fc_no_bias.weight.data = fc.fc1.weight.data


weight_address_ranges = []
for layer in [fc.fc1]:
    # Calculate the range for each layer's weights
    range_size = ceildiv(layer.weight.shape[1] * 4, 64) * 64 * layer.weight.shape[0]
    weight_address_ranges.append(range_size)
weight_start_address_list = [sum(weight_address_ranges[:i]) for i in range(len(weight_address_ranges))]


# --------------------------------------------------
#  Software groundtruth
# --------------------------------------------------
def compute_quantize(model, debug=False, input_data=None):
    """
    Compute the quantized result of the matrix multiplication between the input data and the weight data
    """ 
    # Input
    quantized_input_data = model.fc1.w_high_quantizer(input_data)
    quantized_input_data_str = model.fc1.reconstruct_input(input_data, fixed_point_conversion=True)
    
    # Quantize the weights
    quantized_weight_str = model.fc1.reconstruct_weight(fixed_point_conversion=True) # signed number converted to unsigned
    quantized_bias_str = model.fc1.reconstruct_bias(fixed_point_conversion=True) # signed number converted to unsigned
    quantized_result_str = model.fc1.reconstruct_result(input_data, fixed_point_conversion=True) # signed number converted to unsigned
    quantized_result_scaled_int = model.fc1.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False) # signed number converted to unsigned
        
    no_bias_quantized_result_str = model.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=True) # signed number converted to unsigned
    if debug:
        print("Quantized Weights:")
        for row in quantized_weight_str:
            hex_row = [value for value in row]
            print(hex_row)
        print("Quantized Bias:")
        hex_row = [value for value in quantized_bias_str]
        print(hex_row)
        print("Quantized Input:")
        for row in quantized_input_data:
            hex_row = [value for value in row]
            print(hex_row)
        print("\nQuantized Linear Operation Result:")
        for row in quantized_result_str:
            hex_row = [value for value in row]
            print(hex_row)
        print("\nNo bias quantized result str:")
        for row in no_bias_quantized_result_str:
            hex_row = [value for value in row]
            print(hex_row)
        
    
    weight_size_list = []
    for name, module in model.named_modules():
        # Check if the module is an instance of a linear layer
        if isinstance(module, torch.nn.Linear) or isinstance(module, LinearMixedPrecision):
            print(f"Name: {name}, Shape: {module.weight.shape}")
            if module.bias is not None:
                weight_size_list.append(module.weight.shape)
    
    input_size_list = [input_data.shape]+[(input_data.shape[0], weight_shape[0]) for weight_shape in weight_size_list]
    input_size_list.pop() # last input is accutually the final result 
    
    return input_size_list, weight_size_list, [quantized_result_scaled_int]

ceildiv = lambda a, b: -(-a // b)


async def mixed_precision_test(dut):
    input_size_list, weight_size_list, result_list = compute_quantize(model=fc, debug=True, input_data=input_data)
    combined_list = zip(input_size_list, weight_size_list)
    print(input_size_list, weight_size_list)
    
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
    # Instruct the core
    for layer_index, (input_shape, weight_shape) in enumerate(combined_list):
        byte_per_weight_block = ceildiv(weight_shape[1] * 4, 64) * 64 * systolic_array_size[1]
        byte_per_input_block = ceildiv(input_shape[1] * 4, 64) * 64 * systolic_array_size[0]
        input_matrix_iteration = ceildiv(input_shape[0], systolic_array_size[0])  # number channel blocks
        weight_matrix_iteration = ceildiv(weight_shape[0], systolic_array_size[1]) # number of iteration to produce one output channel blocks
        
        print(f"layer_index={layer_index}")
        print(f"input_shape={input_shape}, weight_shape={weight_shape}")
        print(f"byte_per_weight_block={byte_per_weight_block}, byte_per_input_block={byte_per_input_block}")
        print(f"input_matrix_iteration={input_matrix_iteration}, weight_matrix_iteration={weight_matrix_iteration}")
        
        logger.info("Start executing instructions")
        for i, (writeback_address, offset) in enumerate(writeback_address_generator(writeback_address=feature_start_address_list[layer_index+1], input_matrix_size=input_shape, weight_matrix_size=weight_shape, systolic_array_size=systolic_array_size)):
            reversed_index =  (weight_matrix_iteration-1-(i % weight_matrix_iteration))
            weight_start_address = weight_start_address_list[layer_index] + byte_per_weight_block * reversed_index
            input_start_address = feature_start_address_list[layer_index] + byte_per_input_block * (i // weight_matrix_iteration)
            timeout = max(weight_shape[0] * weight_shape[1], input_shape[0], input_shape[1]) * 100
            logger.info(f"Writeback address: {hex(writeback_address)}, Offset: {offset}, Load weight from: {hex(weight_start_address)}, Load input from: {hex(input_start_address)}")

            await RisingEdge(dut.clk)
            await load_weight_block_instruction_b(dut, start_address=weight_start_address, weight_block_size=weight_shape, timeout=timeout)
            logger.info("Done instructing weight prefetcher")

            await RisingEdge(dut.clk)
            await load_feature_block_instruction_b(dut, start_address=input_start_address, input_block_size=input_shape, timeout=timeout)
            logger.info("Done instructing feature prefetcher")

            await RisingEdge(dut.clk)
            if layer_index==0:
                bias = fc.fc1.reconstruct_bias(fixed_point_conversion=True, str_output=False)
                bias_section = bias[reversed_index*systolic_array_size[1]:(reversed_index+1)*systolic_array_size[1]].tolist()
            elif layer_index==4:
                bias = fc.fc5.reconstruct_bias(fixed_point_conversion=True, str_output=False)
                bias_section = bias[reversed_index*systolic_array_size[1]:(reversed_index+1)*systolic_array_size[1]].tolist()
            else:
                bias_section = None
            logger.info("Feeding bias:")
            logger.info(bias_section)
            Activation_code = Activation.NONE.value if layer_index==0 else Activation.RELU.value
            await calculate_linear_and_writeback_b(dut, writeback_address=writeback_address, offset=offset, output_matrix_size=(input_shape[0], weight_shape[0]), activation_code=Activation_code, timeout=timeout, bias=bias_section)
            logger.info("Done instructing fte")

            await RisingEdge(dut.clk)

        
        # Read the result from the RAM and compare it with the software result
        if layer_index==4:
            software_result_matrix = result_list[layer_index]
            hardware_result_matrix = await read_ram(axi_ram_driver, software_result_matrix, byte_per_feature, feature_start_address_list[layer_index+1])
            logger.info("Hardware matrix:")
            print(hardware_result_matrix)

            logger.info("Software matrix:")
            print(software_result_matrix)
            assert ((hardware_result_matrix == software_result_matrix).all()), "The solution is not correct"
    
    print("reading from:", feature_start_address_list[-1])
    software_result_matrix = result_list[-1]
    hardware_result_matrix = await read_ram(axi_ram_driver, software_result_matrix, byte_per_feature, feature_start_address_list[-1])           
    # Logging results
    logger.info("Hardware matrix:")
    print(hardware_result_matrix)

    logger.info("Software matrix:")
    print(software_result_matrix)
    
    assert ((hardware_result_matrix == software_result_matrix).all()), "The solution is not correct"
            
    
# def test_axi_runner():
#     """
#     Simulate the adder example using the Python runner.
#     This file can be run directly or via pytest discovery.
#     NOTE: This is not needed anymore, as we are primarily using modelsim
#     """
#     sim = "icarus"
#     extra_args = [] #  "--timescale", "1ps/1ps", "-Wno-WIDTH", "-Wno-CASEINCOMPLETE"
#     wave_args = []  #  "--trace-fst", "--trace-structs"
    
#     # equivalent to setting the PYTHONPATH environment variable
#     proj_path = Path(__file__).resolve().parent
#     verilog_sources = [
#         proj_path / "include" / "top_pkg.sv",
#         proj_path / "rtl" / "prefetcher.sv",
#         proj_path / "rtl" / "prefetcher_weight_bank.sv",
#         proj_path / "lib" / "buffer" / "ultraram.v",
#         proj_path / "lib" / "buffer" / "ultraram_fifo.sv",
#         proj_path / "lib" / "axi" / "axi_read_master.sv",
#         proj_path / "lib" / "axi" / "axi_interface.sv",
#         proj_path / "lib" / "axi" / "axi_ram.sv",
#         proj_path / "top.sv",
#     ]

#     runner = get_runner(sim)
#     runner.build(
#         verilog_sources=verilog_sources,
#         hdl_toplevel="top",
#         always=True,
#         build_args=extra_args+wave_args,
#     )
#     runner.test(hdl_toplevel="top", test_module="top_tb")


if __name__ == "__main__":
    # fc = MLP(first_layer_weight_shape)
    
    # Input
    quantized_input_high_data = fc.fc1.w_high_quantizer(input_data)
    quantized_input_high_data_str = fc.fc1.reconstruct_input(input_data, fixed_point_conversion=True)
    quantized_input_high_data_2 = fc.fc1.reconstruct_input(input_data, fixed_point_conversion=False)
    
    quantized_input_low_data = fc.fc1.w_low_quantizer(quantized_input_high_data)
    quantized_input_low_data_str = fc.fc1.reconstruct_input(quantized_input_high_data, high=False, fixed_point_conversion=True)
    quantized_input_low_data_2 = fc.fc1.reconstruct_input(quantized_input_high_data, high=False, fixed_point_conversion=False)
    
    # Quantize the weights
    quantized_weight_str = fc.fc1.reconstruct_weight(fixed_point_conversion=True) # signed number converted to unsigned
    quantized_weight = fc.fc1.reconstruct_weight(fixed_point_conversion=False) # signed number converted to unsigned

    #bias
    quantized_bias_str = fc.fc1.reconstruct_bias(fixed_point_conversion=True) # signed number converted to unsigned
    quantized_bias = fc.fc1.reconstruct_bias(fixed_point_conversion=False, str_output=False) # signed number converted to unsigned
    quantized_bias_scaled_int = fc.fc1.reconstruct_bias(fixed_point_conversion=True, str_output=False) # signed number converted to unsigned
    
    quantized_result_str = fc.fc1.reconstruct_result(input_data, fixed_point_conversion=True) # signed number converted to unsigned
    quantized_result = fc.fc1.reconstruct_result(input_data, fixed_point_conversion=False, str_output=False) 
    quantized_result_scaled_int = fc.fc1.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False) # signed number converted to unsigned
    
    no_bias_quantized_result_str = fc.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=True) # signed number converted to unsigned
    no_bias_quantized_result_scaled_int = fc.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False) # signed number converted to unsigned
    no_bias_quantized_result = fc.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=False, str_output=False) 



    print("Quantized Weights:")
    for row in quantized_weight_str:
        hex_row = [value for value in row]
        print(hex_row)
    print("Transposed Quantized Weight:")
    for row in np.array(quantized_weight_str).T:
        hex_row = [value for value in row]
        print(hex_row)
    print("Quantized Weight Value:")
    for row in quantized_weight.detach().numpy():
        hex_row = [value for value in row]
        print(hex_row)
    print("====================================")
    print("Quantized Bias:")
    hex_row = [value for value in quantized_bias_str]
    print(hex_row)
    print("Quantized Bias Value Scaled int:")
    hex_row = [value for value in quantized_bias_scaled_int]
    print(hex_row)
    print("Quantized Bias Value:")
    hex_row = [value for value in quantized_bias.detach().numpy()]
    print(hex_row)
    print("====================================")
    print("\nQuantized Linear Operation Result:")
    for row in quantized_result_str:
        hex_row = [value for value in row]
        print(hex_row)
    print("\nQuantized Linear Operation Result Scaled Int:")
    for row in quantized_result_scaled_int:
        hex_row = [value for value in row]
        print(hex_row)
    print("\nQuantized Linear Operation Result acctually value:")
    for row in quantized_result.detach().numpy():
        hex_row = [value for value in row]
        print(hex_row)
    
    print("\nNo bias quantized result str:")
    for row in no_bias_quantized_result_str:
        hex_row = [value for value in row]
        print(hex_row)
    print("\nNo bias quantized result scaled int:")
    for row in no_bias_quantized_result_scaled_int:
        hex_row = [value for value in row]
        print(hex_row)
    print("\nNo bias quantized result acctually value:")
    for row in no_bias_quantized_result.detach().numpy():
        hex_row = [value for value in row]
        print(hex_row)
    print("====================================")
    print("Initial Input:")
    for row in input_data.detach().numpy():
        hex_row = [value for value in row]
        print(hex_row)
    print("Quantized Input High:")
    for row in quantized_input_high_data_str:
        hex_row = [value for value in row]
        print(hex_row)
    print("Hight Input Value:")
    for row in quantized_input_high_data.detach().numpy():
        hex_row = [value for value in row]
        print(hex_row)
    print("Quantized Input Low:")
    for row in quantized_input_low_data_str:
        hex_row = [value for value in row]
        print(hex_row)
    print("Low Input Value:")
    for row in quantized_input_low_data.detach().numpy():
        hex_row = [value for value in row]
        print(hex_row)
    print("====================================")
    
    # open("/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", 'w').close()
    write_to_file(quantized_weight_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='w',start_address=weight_start_address_list[0], each_feature_size=4, padding_alignment=64, direct_write_str=True)
    write_to_file(quantized_input_high_data_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=feature_start_address_list[0], each_feature_size=4, padding_alignment=64, direct_write_str=True)
    
    
    print(input_data)
    weight = fc.fc1.w_low_quantizer(fc.fc1.weight)[13, :].detach().numpy()
    low_input = fc.fc1.w_low_quantizer(input_data)[0, :].detach().numpy()
    model_result = fc.fc_no_bias(input_data)
    l_r = torch.nn.functional.linear(fc.fc1.w_low_quantizer(fc.fc1.weight), fc.fc1.w_low_quantizer(input_data))

    r = np.dot(weight, low_input, out=None)
    # print('high input')
    # print(fc.fc1.w_high_quantizer(input_data)[0, :].detach().numpy())
    # print("====================================LOW WEIGHT, LOW INPUT====================================")
    # print(weight, low_input, r)
    # print(l_r)
    # print(model_result)
    
    _list_8 = ['1f', '18', '0e', 'de', '0b', 'ec', 'ff', 'e6', 'f4', '1a', 'fa', 'ea', 'f4', 'f7', 'f4', '0c', '1a', 'fd', 'f8', '07', 'f4', '11', '0d', '1b', '14', '15', '0a', '15', 'fc', '01', 'fc', '0e']
    _list_4 = ['7', '7', '7', '8', '5', '8', '0', '8', 'a', '7', 'd', '8', 'a', 'c', 'a', '6', '7', 'f', 'c', '4', 'a', '7', '6', '7', '7', '7', '5', '7', 'e', '0', 'e', '7']
    binary_list = [bin(int(i, 16))[2:].zfill(8)[::-1] for i in _list_8]
    
    print("====================================")
    print(fc.fc1.w_low_quantizer(torch.tensor(-0.5625)))
    
    # hardware rounding 
    IN_FRAC_WIDTH = 4
    IN_WIDTH = 8
    OUT_FRAC_WIDTH = 3
    OUT_WIDTH = 4
    IN_INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH
    OUT_INT_WIDTH = OUT_WIDTH - OUT_FRAC_WIDTH
    is_one = lambda x: x == '1'