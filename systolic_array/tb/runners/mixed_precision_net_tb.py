# This file offers a testbench for the whole mixed precision neural network
import logging
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
torch.manual_seed(42)  
SYSTOLIC_MODULE_COUNT = 4
systolic_array_size = (1*4, SYSTOLIC_MODULE_COUNT*4)
byte_per_feature, precision = 4, 8

high_precision = (8, 4)
low_precision = (4, 3)

input_shape = (1, 16)
first_layer_weight_shape = (64, 16)
second_layer_weight_shape = (32, 64)
third_layer_weight_shape = (32, 32)
fourth_layer_weight_shape = (32, 32)
fifth_layer_weight_shape = (5, 32)

input_start_address = 0x1100000
weight_start_address = 0x0000000
first_layer_result = 0x1200000
second_layer_result = 0x1300000
third_layer_result = 0x1400000
fourth_layer_result = 0x1500000
fifth_layer_result = 0x1600000

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
class FCN_mix(torch.nn.Module):
    """Fully connected mixed_precision neural network"""
    def __init__(self, first_layer_weight_shape, second_layer_weight_shape, third_layer_weight_shape, fourth_layer_weight_shape, fifth_layer_weight_shape) -> None:
        super().__init__()
        self.fc1 = LinearMixedPrecision(first_layer_weight_shape[1], first_layer_weight_shape[0], bias=True, config=config) # Bias true for the first layer
        self.fc2 = LinearMixedPrecision(second_layer_weight_shape[1], second_layer_weight_shape[0], bias=False, config=config)
        self.fc3 = LinearMixedPrecision(third_layer_weight_shape[1], third_layer_weight_shape[0], bias=False, config=config)
        self.fc4 = LinearMixedPrecision(fourth_layer_weight_shape[1], fourth_layer_weight_shape[0], bias=False, config=config)
        self.fc5 = LinearMixedPrecision(fifth_layer_weight_shape[1], fifth_layer_weight_shape[0], bias=True, config=config) # Bias true for the last layer


    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x).to(torch.int8)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x).to(torch.int8)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x).to(torch.int8)
        x = torch.nn.functional.relu(x)
        x = self.fc4(x).to(torch.int8)
        x = torch.nn.functional.relu(x)
        x = self.fc5(x)
        x = torch.nn.functional.relu(x)
        return x

ceildiv = lambda a, b: -(-a // b)
# --------------------------------------------------
#  Global setting
# --------------------------------------------------

feature_start_address_list = [input_start_address, first_layer_result, second_layer_result, third_layer_result, fourth_layer_result, fifth_layer_result]

input_data = torch.randn(*(input_shape))

fc = FCN_mix(first_layer_weight_shape, second_layer_weight_shape, third_layer_weight_shape, fourth_layer_weight_shape, fifth_layer_weight_shape)
fc.fc1.weight.data = torch.randn(*first_layer_weight_shape) 
fc.fc2.weight.data = torch.randn(*second_layer_weight_shape)
fc.fc3.weight.data = torch.randn(*third_layer_weight_shape)
fc.fc4.weight.data = torch.randn(*fourth_layer_weight_shape)
fc.fc5.weight.data = torch.randn(*fifth_layer_weight_shape)

fc.fc1.bias.data = torch.randn(first_layer_weight_shape[0])
fc.fc5.bias.data = torch.randn(fifth_layer_weight_shape[0])
# fc.fc_no_bias.weight.data = fc.fc1.weight.data

weight_address_ranges = []
for layer in [fc.fc1, fc.fc2, fc.fc3, fc.fc4, fc.fc5]:
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
    quantized_input_data, quantized_input_data_str = model.fc1.w_high_quantizer(input_data), model.fc1.reconstruct_input(input_data, fixed_point_conversion=True)
    
    # Quantize the weights fc1
    fc1_quantized_weight_str = model.fc1.reconstruct_weight(fixed_point_conversion=True)
    fc2_quantized_weight_str = model.fc2.reconstruct_weight(fixed_point_conversion=True)
    fc3_quantized_weight_str = model.fc3.reconstruct_weight(fixed_point_conversion=True)
    fc4_quantized_weight_str = model.fc4.reconstruct_weight(fixed_point_conversion=True)
    fc5_quantized_weight_str = model.fc5.reconstruct_weight(fixed_point_conversion=True)
    quantized_weight_str_list = [fc1_quantized_weight_str, fc2_quantized_weight_str, fc3_quantized_weight_str, fc4_quantized_weight_str, fc5_quantized_weight_str]
    
    
    fc1_quantized_bias_str = model.fc1.reconstruct_bias(fixed_point_conversion=True)
    fc5_quantized_bias_str = model.fc5.reconstruct_bias(fixed_point_conversion=True)
    
    fc1_quantized_result = torch.nn.functional.relu(model.fc1(input_data))
    fc2_quantized_result = torch.nn.functional.relu(model.fc2(fc1_quantized_result))
    fc3_quantized_result = torch.nn.functional.relu(model.fc3(fc2_quantized_result))
    fc4_quantized_result = torch.nn.functional.relu(model.fc4(fc3_quantized_result))
    fc5_quantized_result = model.fc5(fc4_quantized_result)
    
    # result str before relu
    fc1_quantized_result_str = model.fc1.reconstruct_result(input_data, fixed_point_conversion=True)
    fc2_quantized_result_str = model.fc2.reconstruct_result(fc1_quantized_result, fixed_point_conversion=True)
    fc3_quantized_result_str = model.fc3.reconstruct_result(fc2_quantized_result, fixed_point_conversion=True)
    fc4_quantized_result_str = model.fc4.reconstruct_result(fc3_quantized_result, fixed_point_conversion=True)
    fc5_quantized_result_str = model.fc5.reconstruct_result(fc4_quantized_result, fixed_point_conversion=True)
    
    # result scaled int before relu
    # signed_scaled_integer is used for collecting the result after the relu
    fc1_quantized_result_scaled_int = torch.nn.functional.relu(torch.tensor(model.fc1.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False, signed_scaled_integer=True, cast_to_high_precision_format=True))).detach().numpy()
    fc2_quantized_result_scaled_int = torch.nn.functional.relu(torch.tensor(model.fc2.reconstruct_result(fc1_quantized_result, fixed_point_conversion=True, str_output=False, signed_scaled_integer=True, cast_to_high_precision_format=True))).detach().numpy()
    fc3_quantized_result_scaled_int = torch.nn.functional.relu(torch.tensor(model.fc3.reconstruct_result(fc2_quantized_result, fixed_point_conversion=True, str_output=False, signed_scaled_integer=True, cast_to_high_precision_format=True))).detach().numpy()
    fc4_quantized_result_scaled_int = torch.nn.functional.relu(torch.tensor(model.fc4.reconstruct_result(fc3_quantized_result, fixed_point_conversion=True, str_output=False, signed_scaled_integer=True, cast_to_high_precision_format=True))).detach().numpy()
    fc5_quantized_result_scaled_int = torch.nn.functional.relu(torch.tensor(model.fc5.reconstruct_result(fc4_quantized_result, fixed_point_conversion=True, str_output=False, signed_scaled_integer=True, cast_to_high_precision_format=True))).detach().numpy()
    
    # fc2 (no bias)
    result_list_str = [fc1_quantized_result_str, fc2_quantized_result_str, fc3_quantized_result_str, fc4_quantized_result_str, fc5_quantized_result_str]
    result_list_scaled_int = [fc1_quantized_result_scaled_int, fc2_quantized_result_scaled_int, fc3_quantized_result_scaled_int, fc4_quantized_result_scaled_int, fc5_quantized_result_scaled_int]
    
    # no_bias_quantized_result_str = model.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=True)
    if debug:
        print("Quantized Input:")
        for row in quantized_input_data:
            hex_row = [value for value in row]
            print(hex_row)
        
        for i, weight_str in enumerate(quantized_weight_str_list):
            print(f"Quantized Weights_{i}:")
            for row in weight_str:
                hex_row = [value for value in row]
                print(hex_row)
            print(f"Transposed Quantized Weight_{i}:")
            for row in np.array(weight_str).T:
                hex_row = [value for value in row]
                print(hex_row)
            print(f"\n {i} Layer Result before relu:")
            for row in result_list_str[i]:
                hex_row = [value for value in row]
                print(hex_row)
            
    weight_size_list = []
    for name, module in model.named_modules():
        # Check if the module is an instance of a linear layer
        if isinstance(module, torch.nn.Linear) or isinstance(module, LinearMixedPrecision):
            print(f"Name: {name}, Shape: {module.weight.shape}")
            # if module.bias is not None:
            weight_size_list.append(module.weight.shape)
    
    input_size_list = [input_data.shape]+[(input_data.shape[0], weight_shape[0]) for weight_shape in weight_size_list]
    input_size_list.pop() # last input is accutually the final result 
    
    return input_size_list, weight_size_list, result_list_scaled_int

ceildiv = lambda a, b: -(-a // b)

async def mixed_precision_net_test(dut):
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
            Activation_code = Activation.NONE.value if layer_index==10 else Activation.RELU.value
            await calculate_linear_and_writeback_b(dut, writeback_address=writeback_address, offset=offset, output_matrix_size=(input_shape[0], weight_shape[0]), activation_code=Activation_code, timeout=timeout, bias=bias_section)
            logger.info("Done instructing fte")

            await RisingEdge(dut.clk)

        
        # Read the result from the RAM and compare it with the software result
        # if layer_index==4:
        #     software_result_matrix = result_list[layer_index]
        #     hardware_result_matrix = await read_ram(axi_ram_driver, software_result_matrix, byte_per_feature, feature_start_address_list[layer_index+1])
        #     logger.info("Hardware matrix:")
        #     print(hardware_result_matrix)

        #     logger.info("Software matrix:")
        #     print(software_result_matrix)
        #     assert ((hardware_result_matrix == software_result_matrix).all()), "The solution is not correct"
        #     raise
    
    print("reading from:", feature_start_address_list[-1])
    software_result_matrix = result_list[-1]
    hardware_result_matrix = await read_ram(axi_ram_driver, software_result_matrix, byte_per_feature, feature_start_address_list[-1])           
    # Logging results
    logger.info("Hardware matrix:")
    print(hardware_result_matrix)

    logger.info("Software matrix:")
    print(software_result_matrix)
    
    assert ((hardware_result_matrix == software_result_matrix).all()), "The solution is not correct"



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
    fc1_quantized_weight_str = fc.fc1.reconstruct_weight(fixed_point_conversion=True)
    fc2_quantized_weight_str = fc.fc2.reconstruct_weight(fixed_point_conversion=True)
    fc3_quantized_weight_str = fc.fc3.reconstruct_weight(fixed_point_conversion=True)
    fc4_quantized_weight_str = fc.fc4.reconstruct_weight(fixed_point_conversion=True)
    fc5_quantized_weight_str = fc.fc5.reconstruct_weight(fixed_point_conversion=True)
    
    fc1_quantized_weight = fc.fc1.reconstruct_weight(fixed_point_conversion=False)
    
    #bias
    quantized_bias_str = fc.fc1.reconstruct_bias(fixed_point_conversion=True)
    quantized_bias = fc.fc1.reconstruct_bias(fixed_point_conversion=False, str_output=False)
    quantized_bias_scaled_int = fc.fc1.reconstruct_bias(fixed_point_conversion=True, str_output=False)
    
    quantized_result_str = fc.fc1.reconstruct_result(input_data, fixed_point_conversion=True)
    quantized_result = fc.fc1.reconstruct_result(input_data, fixed_point_conversion=False, str_output=False) 
    quantized_result_signed = fc.fc1.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False, signed_scaled_integer=True)
    quantized_result_scaled_int = fc.fc1.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False)
    
    # no_bias_quantized_result_str = fc.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=True)
    # no_bias_quantized_result_scaled_int = fc.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False)
    # no_bias_quantized_result = fc.fc_no_bias.reconstruct_result(input_data, fixed_point_conversion=False, str_output=False) 



    print("Quantized Weights:")
    for row in fc1_quantized_weight_str:
        hex_row = [value for value in row]
        print(hex_row)
    print("Transposed Quantized Weight:")
    for row in np.array(fc1_quantized_weight_str).T:
        hex_row = [value for value in row]
        print(hex_row)
    print("Quantized Weight Value:")
    for row in fc1_quantized_weight.detach().numpy():
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
    print("\nQuantized Linear Operation Result acctually value signed:")
    for row in quantized_result_signed:
        hex_row = [value for value in row]
        print(hex_row)
    print(torch.nn.functional.relu(torch.tensor(quantized_result_signed)))
    print(torch.nn.functional.relu(torch.tensor(fc.fc1.reconstruct_result(input_data, fixed_point_conversion=True, str_output=False, signed_scaled_integer=True, cast_to_high_precision_format=True))).detach().numpy())
    
    # print("\nNo bias quantized result str:")
    # for row in no_bias_quantized_result_str:
    #     hex_row = [value for value in row]
    #     print(hex_row)
    # print("\nNo bias quantized result scaled int:")
    # for row in no_bias_quantized_result_scaled_int:
    #     hex_row = [value for value in row]
    #     print(hex_row)
    # print("\nNo bias quantized result acctually value:")
    # for row in no_bias_quantized_result.detach().numpy():
    #     hex_row = [value for value in row]
    #     print(hex_row)
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
    write_to_file(fc1_quantized_weight_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='w', start_address=weight_start_address_list[0], each_feature_size=4, padding_alignment=64, direct_write_str=True)
    write_to_file(fc2_quantized_weight_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[1], each_feature_size=4, padding_alignment=64, direct_write_str=True)
    write_to_file(fc3_quantized_weight_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[2], each_feature_size=4, padding_alignment=64, direct_write_str=True)
    write_to_file(fc4_quantized_weight_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[3], each_feature_size=4, padding_alignment=64, direct_write_str=True)
    write_to_file(fc5_quantized_weight_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[4], each_feature_size=4, padding_alignment=64, direct_write_str=True)

    write_to_file(quantized_input_high_data_str, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=feature_start_address_list[0], each_feature_size=4, padding_alignment=64, direct_write_str=True)
    
    
    # print(input_data)
    # weight = fc.fc1.w_low_quantizer(fc.fc1.weight)[13, :].detach().numpy()
    # low_input = fc.fc1.w_low_quantizer(input_data)[0, :].detach().numpy()
    # model_result = fc.fc_no_bias(input_data)
    # l_r = torch.nn.functional.linear(fc.fc1.w_low_quantizer(fc.fc1.weight), fc.fc1.w_low_quantizer(input_data))

    # r = np.dot(weight, low_input, out=None)
    # print('high input')
    # print(fc.fc1.w_high_quantizer(input_data)[0, :].detach().numpy())
    # print("====================================LOW WEIGHT, LOW INPUT====================================")
    # print(weight, low_input, r)
    # print(l_r)
    # print(model_result)
    

    # print("====================================")
    # print(fc.fc1.w_low_quantizer(torch.tensor(-0.5625)))