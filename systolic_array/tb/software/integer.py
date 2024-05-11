from math import ceil, log2

from numpy import ndarray, vectorize
from torch import Tensor
import torch

from .utils import my_clamp, my_round


def _integer_quantize(
    x: Tensor | ndarray, width: int, frac_width: int, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    if is_signed:
        int_min = -(2 ** (width - 1))
        int_max = 2 ** (width - 1) - 1
    else:
        int_min = 0
        int_max = 2**width - 1
    # thresh = 2 ** (width - 1)
    scale = 2**frac_width

    if isinstance(x, (Tensor, ndarray)):
        return my_clamp(my_round(x.mul(scale)), int_min, int_max).div(scale)
    elif isinstance(x, int):
        return x
    else:
        return my_clamp(my_round(x * scale), int_min, int_max) / scale


class IntegerQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, width: int, frac_width: int, is_signed: bool = True):
        return _integer_quantize(
            x, width=width, frac_width=frac_width, is_signed=is_signed
        )

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None


def integer_quantizer(
    x: Tensor | ndarray, width: int, frac_width: int, is_signed: bool = True
):
    """
    - Do linear quantization to input according to a scale and number of bits
    - Note that `bias` can be negative or larger than `bits`

    ---
    - forward: convert IEEE FP32/64 to fixed-point
    - backward: STE

    ---
    width: the bit width of the fixed-point number
    frac_width: the number of fractional bits. Note that `bias` can be negative or larger than `bits`

    ---
    For example: 0b101 . 00111, bits = 8, bias = 5

    """
    return IntegerQuantize.apply(x, width, frac_width, is_signed)


def integer_fraction(
    width: int, frac_choices: list, min_value: float, max_value: float
):
    max_half_range = max(abs(min_value), abs(max_value))
    int_width = int(log2(max(0.5, max_half_range))) + 2
    frac_width = max(0, width - int_width)
    frac_width = max(filter(lambda x: x <= frac_width, frac_choices))
    return frac_width

def _integer_hex_str(scaled_up_integer: int, width: int, str_output: bool = False) -> str | int:
    """
    Convert the integer to hex string

    :param scaled_up_integer: The target integer
    :type scaled_up_integer: int
    :param width: The bit width of the integer
    :type width: int
    :param str_output: If True, return the hex string or return the unsigned integer defaults to False, defaults to False
    :type str_output: bool, optional
    :return: The hex string or the unsigned integer
    :rtype: str | int
    """
    converted_unsigned = int(scaled_up_integer) +2**32
    _bin_str = bin(converted_unsigned)[-width:]
    _hex_str = hex(int(_bin_str, 2))[2:]
    padded_zero_to_32_bit = '0'*(8-len(_hex_str)) + _hex_str
    if str_output:
        return padded_zero_to_32_bit
    else:
        return int(padded_zero_to_32_bit, 16)
    
def integer_quantize_in_hex(
        x, width: int, frac_width: int, is_signed: bool = True, str_output: bool = True
    ):
        """
        Quantize the input tensor to fixed-point integer and return the hex string or the unsigned integer

        :param x: The input tensor
        :type x: Tensor | ndarray | int | float
        :param width: The bit width of the integer
        :type width: int
        :param frac_width: The fractional bit width
        :type frac_width: int
        :param is_signed: If True, the integer is signed, defaults to True, defaults to True
        :type is_signed: bool, optional
        :param str_output: If True, return the hex string or return the unsigned integer defaults to True, defaults to True
        :type str_output: bool, optional
        :return: The ndarray of hex string or of the unsigned integer. Or the hex string or the unsigned integer for input is not a tensor
        :rtype: ndarray | str | int
        """
        if is_signed:
            int_min = -(2 ** (width - 1))
            int_max = 2 ** (width - 1) - 1
        else:
            int_min = 0
            int_max = 2**width - 1

        scale = 2**frac_width
        f = lambda x: _integer_hex_str(x, width, str_output)
        
        if isinstance(x, (Tensor, ndarray)):
            scaled_up_integer =  ((x.mul(scale)).round()).clamp(int_min, int_max)
            mask = vectorize(f)(scaled_up_integer)
            return mask
        else:
            scaled_up_integer = (((x * scale)).round()).clip(int_min, int_max)
            _bin_str = _integer_hex_str(scaled_up_integer, width, str_output)
            return _bin_str
