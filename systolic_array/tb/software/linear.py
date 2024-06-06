import math
from functools import partial
from math import ceil, log2

import torch
from torch import Tensor
from torch.nn import functional as F
from numpy import full 

from .integer import (
    integer_quantizer,
    integer_quantize_in_hex,
)

class _LinearBase(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.bypass = False
        self.x_quantizer = None
        self.w_quantizer = None
        self.b_quantizer = None
        self.pruning_masks = None

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            # if bypss, there is no quantization
            return F.linear(x, self.weight, self.bias)
        else:
            x = self.x_quantizer(x)
            w = self.w_quantizer(self.weight)
            bias = self.b_quantizer(self.bias) if self.bias is not None else None
            return F.linear(x, w, bias)


class LinearInteger(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizer
        w_width, w_frac_width = config["weight_width"], config["weight_frac_width"]
        x_width, x_frac_width = config["data_in_width"], config["data_in_frac_width"]
        # check bias quantizer, if not, use weight quantizer
        b_width, b_frac_width = config["bias_width"], config["bias_frac_width"]
        self.w_quantizer = partial(
            integer_quantizer, width=w_width, frac_width=w_frac_width
        )
        self.x_quantizer = partial(
            integer_quantizer, width=x_width, frac_width=x_frac_width
        )
        self.b_quantizer = partial(
            integer_quantizer, width=b_width, frac_width=b_frac_width
        )
class LinearMixedPrecision(_LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        assert config is not None, "config is None!"
        self.config = config
        self.bypass = config.get("bypass", False)
        if self.bypass:
            return
        # establish quantizer
        w_high_width, w_high_frac_width = config["weight_high_width"], config["weight_high_frac_width"]
        w_low_width, w_low_frac_width = config["weight_low_width"], config["weight_low_frac_width"]
        
        x_high_width, x_high_frac_width = config["data_in_high_width"], config["data_in_high_frac_width"]
        x_low_width, x_low_frac_width = config["data_in_low_width"], config["data_in_low_frac_width"]
        
        b_high_width, b_high_frac_width = config["bias_high_width"], config["bias_high_frac_width"]
        b_low_width, b_low_frac_width = config["bias_low_width"], config["bias_low_frac_width"]
        
        self.block_width, self.block_height = config["block_width"], config["block_height"]
        self.block_high_width = config["block_high_width"]
        
        self.w_high_quantizer = partial(
            integer_quantizer, width=w_high_width, frac_width=w_high_frac_width
        )
        self.w_low_quantizer = partial(
            integer_quantizer, width=w_low_width, frac_width=w_low_frac_width
        )
        self.x_high_quantizer = partial(
            integer_quantizer, width=x_high_width, frac_width=x_high_frac_width
        )
        self.x_low_quantizer = partial(
            integer_quantizer, width=x_low_width, frac_width=x_low_frac_width
        )
        self.b_high_quantizer = partial(
            integer_quantizer, width=b_high_width, frac_width=b_high_frac_width
        )
        self.b_low_quantizer = partial(
            integer_quantizer, width=b_low_width, frac_width=b_low_frac_width
        )
            
    def reconstruct_weight(self, fixed_point_conversion=False):
        """
        Reconstruct the weight tensor with mixed-precision fixed-point representation
        E.g 0b101 . 00111
        Result will be padded to 32 bits. Extra not valid bit in the front will be padded with zero
        
        :param fixed_point_conversion: If True, return the converted fixed-point representation or just the weight value defaults to False
        :type fixed_point_conversion: bool, optional
        """
        w_high = self.w_high_quantizer(self.weight)
        w_low = self.w_low_quantizer(self.weight)
        block_per_row = math.ceil(self.out_features / self.block_width)
        
        if fixed_point_conversion:
            w_high = w_high.detach()
            w_low = w_low.detach()
            w_high = integer_quantize_in_hex(w_high, self.config['weight_high_width'],self.config['weight_high_frac_width'])
            w_low = integer_quantize_in_hex(w_low, self.config['weight_low_width'],self.config['weight_low_frac_width'])
            constructed_weight = full(self.weight.shape, '0'*self.config['weight_high_width'])
        else:
            constructed_weight = torch.Tensor(*self.weight.shape)
            
        for j in range(block_per_row):
            col_end = min((j + 1) * self.block_width, self.out_features)
            constructed_weight[j * self.block_width : col_end, :] = w_high[j * self.block_width : col_end, :]
            constructed_weight[j * self.block_width + self.block_high_width : col_end, :] = w_low[j * self.block_width + self.block_high_width : col_end, :]
        
        return constructed_weight
        
    def reconstruct_bias(self, fixed_point_conversion=False, str_output=True):
        """
        Reconstruct the bias tensor with mixed-precision fixed-point representation
        E.g 0b101 . 00111
        Result will be padded to 32 bits. Extra not valid bit in the front will be padded with zero
        
        fixed_point_conversion: return the converted fixed-point representation
        string_output: return the fixed-point representation in hex string or returned as number
        
        :param fixed_point_conversion: If True, return the converted fixed-point representation or just the weight value defaults to False
        :type fixed_point_conversion: bool, optional
        :param str_output: If True, return the hex string or return the unsigned integer defaults to True, defaults to True
        :type str_output: bool, optional
        """
        if self.bias is None:
            return None
        b_high = self.b_high_quantizer(self.bias)
        b_low = self.b_low_quantizer(self.bias)
        block_per_row = math.ceil(self.out_features / self.block_width)
        
        if fixed_point_conversion:
            b_high = b_high.detach()
            b_low = b_low.detach()
            b_high = integer_quantize_in_hex(b_high, self.config['bias_high_width'],self.config['bias_high_frac_width'], is_signed=True, str_output=str_output)
            b_low = integer_quantize_in_hex(b_low, self.config['bias_low_width'],self.config['bias_low_frac_width'], is_signed=True, str_output=str_output)
            if str_output:
                constructed_bias = full((self.weight.shape[0],), '0'*self.config['bias_high_width'])
            else:
                constructed_bias = full((self.weight.shape[0],), 0)
        else:
            constructed_bias = torch.zeros(self.weight.shape[0],)
            
        for j in range(block_per_row):
            col_end = min((j + 1) * self.block_width, self.out_features)
            constructed_bias[j * self.block_width : col_end] = b_high[j * self.block_width : col_end]
            constructed_bias[j * self.block_width + self.block_high_width : col_end] = b_low[j * self.block_width + self.block_high_width : col_end]
            
        return constructed_bias
    
    def reconstruct_input(self, x, high=True, fixed_point_conversion=False):
        """
        Reconstruct the input tensor with high precision fixed-point representation
        E.g 0b101 . 00111
        Result will be padded to 32 bits. Extra not valid bit in the front will be padded with zero
        
        :param x: The input tensor
        :type x: Tensor | ndarray | int | float
        :param fixed_point_conversion: If True, return the converted fixed-point representation or just the weight value defaults to False
        :type fixed_point_conversion: bool, optional
        """
        if high:
            x_high = self.x_high_quantizer(x)
            if fixed_point_conversion:
                return integer_quantize_in_hex(x, self.config['data_in_high_width'],self.config['data_in_high_frac_width'])
            else:
                return x_high
        else:
            x_high = self.x_low_quantizer(x)
            if fixed_point_conversion:
                return integer_quantize_in_hex(x, self.config['data_in_low_width'],self.config['data_in_low_frac_width'])
            else:
                return x_high
        
            
    
    def reconstruct_result(self, x, fixed_point_conversion=False, str_output=True, signed_scaled_integer=False, cast_to_high_precision_format=False):
        """
        Reconstruct the input tensor with high precision fixed-point representation
        E.g 0b101 . 00111
        Result will be padded to 32 bits. Extra not valid bit in the front will be padded with zero
        
        :param x: The input tensor
        :type x: Tensor | ndarray | int | float
        :param fixed_point_conversion: If True, return the converted fixed-point representation or just the weight value defaults to False
        :type fixed_point_conversion: bool, optional
        :param str_output: If True, return the hex string or return the unsigned scaled integer defaults to True, defaults to True
        :type str_output: bool, optional
        :param signed_scaled_integer: If True, return the signed integer, defaults to False
        """
        w_high = self.w_high_quantizer(self.weight)
        w_low = self.w_low_quantizer(self.weight)
        x_high = self.x_high_quantizer(x)
        x_low = self.x_low_quantizer(x_high)

        if self.bias is not None:
            b_high = self.b_high_quantizer(self.bias)
            b_low = self.b_low_quantizer(self.bias)
        else:
            b_high = None
            b_low = None
            
        result_high = F.linear(x_high, w_high, b_high)
        result_low = F.linear(x_low, w_low, b_low)
        
        if fixed_point_conversion:
            result_high = result_high.detach()
            result_low = result_low.detach()
            result_high = integer_quantize_in_hex(result_high, self.config['weight_high_width'],self.config['weight_high_frac_width'], is_signed=True, str_output=str_output, signed_scaled_integer=signed_scaled_integer)
            if cast_to_high_precision_format:
                result_low = integer_quantize_in_hex(result_low, self.config['weight_high_width'],self.config['weight_high_frac_width'], is_signed=True, str_output=str_output, signed_scaled_integer=signed_scaled_integer)
            else:
                result_low = integer_quantize_in_hex(result_low, self.config['weight_low_width'],self.config['weight_low_frac_width'], is_signed=True, str_output=str_output, signed_scaled_integer=signed_scaled_integer)
            constructed_result = full(result_high.shape, '0'*self.config['weight_high_width'])
            if str_output:
                constructed_result = full(result_high.shape, '0'*self.config['weight_high_width'])
            else:
                constructed_result = full(result_high.shape, 0)
        else:
            constructed_result = torch.Tensor(*result_high.shape)
            
        block_per_row = math.ceil(self.out_features / self.block_width)
            
        for j in range(block_per_row):
            col_end = min((j + 1) * self.block_width, self.out_features)
            constructed_result[:, j * self.block_width : col_end] = result_high[:, j * self.block_width : col_end]
            constructed_result[:, j * self.block_width + self.block_high_width : col_end] = result_low[:, j * self.block_width + self.block_high_width : col_end]
    
        return constructed_result
        
    def forward(self, x: Tensor) -> Tensor:
        w_high = self.w_high_quantizer(self.weight)
        w_low = self.w_low_quantizer(self.weight)
        x_high = self.x_high_quantizer(x)
        x_low = self.x_low_quantizer(x_high)

        if self.bias is not None:
            b_high = self.b_high_quantizer(self.bias)
            b_low = self.b_low_quantizer(self.bias)
        else:
            b_high = None
            b_low = None
        
        result_high = F.linear(x_high, w_high, b_high)
        result_low = F.linear(x_low, w_low, b_low)
        
        batch_size, _ = x.shape
        result = torch.zeros(batch_size, self.out_features).to(x.device)
        block_per_row = math.ceil(self.out_features / self.block_width)
        block_per_col = math.ceil(batch_size / self.block_height)
        
        for j in range(block_per_row):
            col_end = min((j + 1) * self.block_width, self.out_features)
            
            result[:, j * self.block_width : col_end] = result_high[:, j * self.block_width : col_end]
            result[:, j * self.block_width + self.block_high_width : col_end] = result_low[:, j * self.block_width + self.block_high_width : col_end]
        return result