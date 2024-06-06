# This file offers a simple test for an AXI ram module
import logging
import os
import sys
from pathlib import Path
from math import ceil, log2

from numpy import ndarray
from torch import Tensor
import torch

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
from systolic_array.tb.software.instruction import calculate_linear_and_writeback_b, load_feature_block_instruction_b, load_weight_block_instruction_b, clear_all
from systolic_array.tb.software.ram import write_to_file, writeback_address_generator
from systolic_array.tb.software.linear import LinearInteger, LinearMixedPrecision


config = {
    # High precision
    "weight_high_width": 8,
    "weight_high_frac_width": 4,
    "data_in_high_width": 8,
    "data_in_high_frac_width": 4,
    "bias_high_width": 8,
    "bias_high_frac_width": 4,
    # Low precision
    "weight_low_width": 4,
    "weight_low_frac_width": 1,
    "data_in_low_width": 4,
    "data_in_low_frac_width": 1,
    "bias_low_width": 4,
    "bias_low_frac_width": 1,
    #
    "block_width": 8,
    "block_height": 4,
    "block_high_width": 4,
    # 
    "weight_width": 8,
    "weight_frac_width": 4,
    "data_in_width": 8,
    "data_in_frac_width": 4,
    "bias_width": 8,
    "bias_frac_width": 4,
    "bypass": False,
}
torch.manual_seed(42) 
layer = LinearMixedPrecision(8, 8, bias=False, config=config)
layer1 = LinearInteger(8, 8, bias=False, config=config)
weight_matrix = torch.randn(8, 8)
input_matrix = torch.randn(4, 8)
layer.weight.data = weight_matrix
layer1.weight.data = weight_matrix
# print(layer1.w_quantizer(layer1.weight))
# print(layer.reconstruct_weight())
# print("--"*50)
# print(layer(input_matrix))
# # print(input_matrix@layer.reconstruct_weight().transpose(0, 1))
# print(layer1(input_matrix))


def _integer_bin_str(scaled_up_integer):
    scaled_up_integer = int(scaled_up_integer)
    _bin_str = hex(scaled_up_integer)[2:]
    return _bin_str
    
def integer_quantize_in_hex(
    x, width: int, frac_width: int, is_signed: bool = True
):
    """
    Return the hex string of the quantized number
    """
    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1

    scale = 2**frac_width
    f = lambda x: _integer_bin_str(x)
    
    if isinstance(x, (Tensor, ndarray)):
        scaled_up_integer =  ((x.mul(scale)).round()).clamp(int_min, int_max)
        mask = np.vectorize(f)(scaled_up_integer)
        return mask
    else:
        scaled_up_integer = (((x * scale)).round()).clip(int_min, int_max)
        _bin_str = _integer_bin_str(scaled_up_integer)
        return _bin_str
    
a = torch.tensor([10.234, 10.234, 10.234])
a = integer_quantize_in_hex(a, 4,2)