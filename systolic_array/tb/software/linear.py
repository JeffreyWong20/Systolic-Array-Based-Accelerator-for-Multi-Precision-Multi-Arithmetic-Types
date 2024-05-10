import math
from functools import partial
from math import ceil, log2

import torch
from torch import Tensor
from torch.nn import functional as F

from .integer import (
    integer_quantizer,
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

    def forward(self, x: Tensor) -> Tensor:
        w_high = self.w_high_quantizer(self.weight)
        w_low = self.w_low_quantizer(self.weight)
        x_high = self.x_high_quantizer(x)
        x_low = self.x_low_quantizer(x)

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