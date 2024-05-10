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
        "..",
    )
)
from systolic_array.tb.software.axi_driver import AXIDriver
from systolic_array.tb.software.instruction import calculate_linear_and_writeback_b, load_feature_block_instruction_b, load_weight_block_instruction_b, clear_all, Activation
from systolic_array.tb.software.ram import read_ram, write_to_file, writeback_address_generator
from systolic_array.tb.software.linear import LinearInteger
np.set_printoptions(threshold=sys.maxsize)

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
class FCN(nn.Module):
    """Fully connected neural network"""
    def __init__(self, first_layer_weight_shape, second_layer_weight_shape, third_layer_weight_shape) -> None:
        super().__init__()
        self.fc1 = LinearInteger(first_layer_weight_shape[0], first_layer_weight_shape[1], bias=True, config=config) # Bias true for the first layer
        self.fc2 = LinearInteger(second_layer_weight_shape[0], second_layer_weight_shape[1], bias=False, config=config)
        self.fc3 = LinearInteger(third_layer_weight_shape[0], third_layer_weight_shape[1], bias=False, config=config)
        self.fc4 = LinearInteger(fourth_layer_weight_shape[0], fourth_layer_weight_shape[1], bias=False, config=config)
        self.fc5 = LinearInteger(fifth_layer_weight_shape[0], fifth_layer_weight_shape[1], bias=True, config=config) # Bias true for the last layer

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
        return x

class JSC_S(nn.Module):
    def __init__(self, info):
        super(JSC_S, self).__init__()
        self.config = info
        self.num_features = 16
        self.num_classes = 5
        hidden_layers = [64, 32, 32, 32]
        self.num_neurons = [self.num_features] + hidden_layers + [self.num_classes]
        layer_list = []
        for i in range(1, len(self.num_neurons)):
            in_features = self.num_neurons[i - 1]
            out_features = self.num_neurons[i]
            bn = nn.BatchNorm1d(out_features)
            layer = []
            if i == 1:
                bn_in = nn.BatchNorm1d(in_features)
                in_act = nn.ReLU()
                fc = LinearInteger(in_features, out_features, bias=False, config=config)
                out_act = nn.ReLU()
                layer = [bn_in, in_act, fc, bn, out_act]
            elif i == len(self.num_neurons) - 1:
                fc = LinearInteger(in_features, out_features, bias=False, config=config)
                out_act = nn.ReLU()
                layer = [fc, bn, out_act]
            else:
                fc = LinearInteger(in_features, out_features, bias=False, config=config)
                out_act = nn.ReLU()
                layer = [fc, out_act]
            layer_list = layer_list + layer
        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for l in self.module_list:
            x = l(x)
        return x

ceildiv = lambda a, b: -(-a // b)

# --------------------------------------------------
#  Global setting
# --------------------------------------------------
torch.manual_seed(42)  
systolic_array_size = (1*4, 4*4)
byte_per_feature, precision = 4, 8

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

feature_start_address_list = [input_start_address, first_layer_result, second_layer_result, third_layer_result, fourth_layer_result, fifth_layer_result]

input_data = torch.randint(0, 128, size=(1, 16), dtype=torch.float32)

fc = FCN(first_layer_weight_shape, second_layer_weight_shape, third_layer_weight_shape)
fc.fc1.weight.data = torch.randint(-128, 127, size=first_layer_weight_shape, dtype=torch.float32)   
fc.fc2.weight.data = torch.randint(-128, 127, size=second_layer_weight_shape, dtype=torch.float32)   
fc.fc3.weight.data = torch.randint(-128, 127, size=third_layer_weight_shape, dtype=torch.float32)
fc.fc4.weight.data = torch.randint(-128, 127, size=fourth_layer_weight_shape, dtype=torch.float32)
fc.fc5.weight.data = torch.randint(-128, 127, size=fifth_layer_weight_shape, dtype=torch.float32)

fc.fc1.bias.data = torch.randint(-128, 127, size=(first_layer_weight_shape[0],), dtype=torch.float32)
fc.fc5.bias.data = torch.randint(-128, 127, size=(fifth_layer_weight_shape[0],), dtype=torch.float32)
print("first layer bias:", fc.fc1.bias.data.numpy().astype(np.uint8).tolist())
print("last layer bias:", fc.fc5.bias.data.numpy().astype(np.uint8).tolist())

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
    Args:
        debug (bool): print the intermediate results
    """

    # Initial Input
    quantized_input_data = model.fc1.w_quantizer(input_data).numpy().astype(np.uint8)
    
    # Quantize the weights
    quantized_weight = [model.fc1.w_quantizer(model.fc1.weight.data), model.fc2.w_quantizer(model.fc2.weight.data), model.fc3.w_quantizer(model.fc3.weight.data), model.fc4.w_quantizer(model.fc4.weight.data), model.fc5.w_quantizer(model.fc5.weight.data)]
    quantized_weight_list = [weight.numpy().astype(np.int8) for weight in quantized_weight]
    
    # Perform the matrix multiplication
    first_layer_result = torch.nn.functional.relu(model.fc1(input_data).to(torch.int8))
    second_layer_result = torch.nn.functional.relu(model.fc2(first_layer_result).to(torch.int8))
    third_layer_result = torch.nn.functional.relu(model.fc3(second_layer_result).to(torch.int8))
    fourth_layer_result = torch.nn.functional.relu(model.fc4(third_layer_result).to(torch.int8))
    fifth_layer_result = model.fc5(fourth_layer_result).to(torch.int8)
    result_list = [first_layer_result, second_layer_result, third_layer_result, fourth_layer_result, fifth_layer_result]
    
    # Model Final Result 
    int_8_result = model(input_data).detach().numpy().astype(np.uint8)
    

    weight_size_list = []
    for name, module in model.named_modules():
        # Check if the module is an instance of a linear layer
        if isinstance(module, torch.nn.Linear) or isinstance(module, LinearInteger):
            print(f"Name: {name}, Shape: {module.weight.shape}")
            weight_size_list.append(module.weight.shape)
    
    input_size_list = [input_data.shape]+[(input_data.shape[0], weight_shape[0]) for weight_shape in weight_size_list]
    input_size_list.pop() # last input is accutually the final result 


    if debug:
        print("Quantized Input:")
        for row in quantized_input_data:
            hex_row = [hex(value) for value in row]
            print(hex_row)
        for i, weight in enumerate(quantized_weight_list):
            print(f"Quantized Weights_{i}")
            uint_weight = weight.astype(np.uint8)
            for row in uint_weight:
                hex_row = [hex(value) for value in row]
                print(hex_row)
            print(f"\nTransposed Quantized Weights_{i}:")
            transposed_weight = weight.transpose().astype(np.uint8)
            for row in transposed_weight:
                hex_row = [hex(value) for value in row]
                print(hex_row)
            print(f"\n {i} Layer Result from model:")
            for row in result_list[i]:
                numpy_version = row.numpy().astype(np.uint8)
                hex_row = [hex(value) for value in numpy_version]
                print(hex_row)
        print("\nMatrix Multiplication Result:")
        for row in int_8_result:
            hex_row = [hex(value) for value in row]
            print(hex_row)
        
    return input_size_list, weight_size_list, [layer_result.numpy().astype(np.uint8) for layer_result in result_list]+[int_8_result]

# --------------------------------------------------
async def fcn_test(dut):
    input_size_list, weight_size_list, result_list = compute_quantize(model=fc, debug=True, input_data=input_data)
    combined_list = zip(input_size_list, weight_size_list)
    print(combined_list)
    
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
                bias = fc.fc1.bias.data.numpy().astype(np.uint8).tolist()
                bias_section = bias[reversed_index*systolic_array_size[1]:(reversed_index+1)*systolic_array_size[1]]
            elif layer_index==4:
                bias = fc.fc5.bias.data.numpy().astype(np.uint8).tolist()
                bias_section = bias[reversed_index*systolic_array_size[1]:(reversed_index+1)*systolic_array_size[1]]
            else:
                bias_section = None
            logger.info("Feeding bias:")
            logger.info(bias_section)
            Activation_code = Activation.NONE.value if layer_index==4 else Activation.RELU.value
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
    
    # Assert the equality of hardware and software matrices
    logger.info("Comparing hardware and software matrices...")
    assert ((hardware_result_matrix == software_result_matrix).all()), "The solution is not correct"
    logger.info("Matrices match: Hardware and software solutions are equivalent.")
        

if __name__ == "__main__":    
    for i, (writeback_address, offset) in enumerate(writeback_address_generator(writeback_address=feature_start_address_list[-1], input_matrix_size=input_data.shape, weight_matrix_size=fc.fc1.weight.shape, systolic_array_size=systolic_array_size)):
        print("input_matrix_size", input_data.shape)
        print("weight_matrix_size", fc.fc1.weight.shape)
        print("systolic_array_size", systolic_array_size)
        print(hex(writeback_address), offset)
   
 
    open("/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", 'w').close()
    # weight
    write_to_file(fc.fc1.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='w', start_address=weight_start_address_list[0], each_feature_size=4, padding_alignment=64)
    write_to_file(fc.fc2.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[1], each_feature_size=4, padding_alignment=64)
    write_to_file(fc.fc3.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[2], each_feature_size=4, padding_alignment=64)
    write_to_file(fc.fc4.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[3], each_feature_size=4, padding_alignment=64)
    write_to_file(fc.fc5.weight.data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=weight_start_address_list[4], each_feature_size=4, padding_alignment=64)
    # input
    write_to_file(input_data, "/home/thw20/FYP/systolic_array/hw/sim/cocotb/weight.mem", writing_mode='a', start_address=feature_start_address_list[0], each_feature_size=4, padding_alignment=64)
    