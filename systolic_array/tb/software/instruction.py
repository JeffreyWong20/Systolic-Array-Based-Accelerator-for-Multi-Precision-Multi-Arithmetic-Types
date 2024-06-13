from cocotb.triggers import RisingEdge, Timer, ReadOnly, ReadWrite, FallingEdge
import logging

debug = True
# create a logger client
logger = logging.getLogger("instruction")
if debug:
    logger.setLevel(logging.DEBUG)

from enum import Enum

# Define an enumeration class
class Activation(Enum):
    NONE = 0
    RELU = 1
    LEAKT_RELU = 2
    RESERVED_ACTIVATION = 3
    
# --------------------------------------------------
# Instruction
# --------------------------------------------------
def load_weight_block_instruction(dut, start_address=0x0000, weight_block_size=(4, 128), precision=1):
    dut.weight_prefetcher_req_valid.value           = 1                         # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value      = 0                         # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value   = start_address             # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value     = weight_block_size[1]     # number of input features                     
    dut.weight_prefetcher_req.out_features.value    = weight_block_size[0]     # number of output features
    dut.weight_prefetcher_req.nodeslot.value        = 0                         # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = precision              # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0                         # not used for weight bank requests

def load_feature_block_instruction(dut, start_address=0x0000, input_block_size=(8, 128), precision=1):
    dut.feature_prefetcher_req_valid.value          = 1                         # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value     = 0                         # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = start_address             # start address of the feature bank
    dut.feature_prefetcher_req.in_features.value    = input_block_size[1]      # number of input features
    dut.feature_prefetcher_req.out_features.value   = input_block_size[0]      # number of output features
    dut.feature_prefetcher_req.nodeslot.value       = 0                         # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = precision             # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0                        # not used for weight bank requests
    
def calculate_linear_and_writeback(dut, writeback_address=0x200000000, output_matrix_size=(8, 8), offset=0, precision=1):
    dut.nsb_fte_req_valid.value = 1                                             # enable the fte
    dut.nsb_fte_req.precision.value = 1                                         # 01 is for fixed 8-bit precision
    dut.layer_config_out_channel_count.value = output_matrix_size[0]            # here we used the first dimension of the input matrix as output channel count
    dut.layer_config_out_features_count.value = output_matrix_size[1]           # here we used the first dimension of the weight matrix as output features count       
    dut.layer_config_out_features_address_msb_value.value = (writeback_address >> 32) & 0b11        # 2 is for the msb of 34 bits address
    dut.layer_config_out_features_address_lsb_value.value = writeback_address & 0xFFFFFFFF          # 0 for the rest of the address
    dut.writeback_offset.value = offset                                         # 0 for the writeback offset
    
    
# --------------------------------------------------
# reset prefetcher
# --------------------------------------------------
async def clear_all(dut):
    dut.weight_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.weight_prefetcher_req.out_features.value = 0                 # number of output features
    dut.weight_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests
    dut.feature_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.feature_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.feature_prefetcher_req.out_features.value = 0                 # number of output features
    dut.feature_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests
    dut.nsb_fte_req_valid.value = 0
    dut.nsb_fte_req.precision.value = 0
    dut.nsb_fte_req.nodeslots.value = 0
    await RisingEdge(dut.clk)
    
async def clear_nsb_prefetcher(dut):
    dut.weight_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.weight_prefetcher_req.out_features.value = 0                 # number of output features
    dut.weight_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests
    dut.feature_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.feature_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.feature_prefetcher_req.out_features.value = 0                 # number of output features
    dut.feature_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests
    await RisingEdge(dut.clk)

async def clear_weight_prefetcher(dut):
    dut.weight_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.weight_prefetcher_req.out_features.value = 0                 # number of output features
    dut.weight_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests
    await RisingEdge(dut.clk)

async def clear_feature_prefetcher(dut):
    dut.feature_prefetcher_req_valid.value = 0                        # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value   = 0                 # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = 0x0000          # start address of the weight bank
    dut.feature_prefetcher_req.in_features.value  = 0                 # number of input features
    dut.feature_prefetcher_req.out_features.value = 0                 # number of output features
    dut.feature_prefetcher_req.nodeslot.value     = 0                 # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = 0           # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0              # not used for weight bank requests
    await RisingEdge(dut.clk)
    
# --------------------------------------------------
# reset fte
# --------------------------------------------------
async def clear_fte(dut):
    dut.nsb_fte_req_valid.value = 0
    dut.nsb_fte_req.precision.value = 0
    dut.nsb_fte_req.nodeslots.value = 0
    await RisingEdge(dut.clk)
    
# --------------------------------------------------
# Blocking Instruction
# --------------------------------------------------
async def load_weight_block_instruction_b(dut, start_address=0x0000, weight_block_size=(4, 128), precision=1, blocking=True, timeout=10000):
    dut.weight_prefetcher_req_valid.value           = 1                         # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value      = 0                         # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value   = start_address             # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value     = weight_block_size[1]     # number of input features                     
    dut.weight_prefetcher_req.out_features.value    = weight_block_size[0]     # number of output features
    dut.weight_prefetcher_req.nodeslot.value        = 0                         # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = precision              # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0                         # not used for weight bank requests
    number_of_clock = 1 
    await RisingEdge(dut.clk)
    if blocking:
        p = 0
        while True:
            number_of_clock += 1
            await FallingEdge(dut.clk)
            await ReadOnly()
            if dut.weight_prefetcher_resp_valid.value == 1:
                logger.info("Weight prefetcher response is valid")
                break
            elif p==timeout:
                raise ValueError("Deadlock detected: weight_prefetcher_req_ready are not ready")
            p+=1
    await RisingEdge(dut.clk)
    await clear_weight_prefetcher(dut)
    logger.info("Weight prefetcher is reset")
    return number_of_clock
        
async def load_feature_block_instruction_b(dut, start_address=0x0000, input_block_size=(8, 128), precision=1, blocking=True, timeout=10000):
    dut.feature_prefetcher_req_valid.value          = 1                         # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value     = 0                         # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = start_address             # start address of the feature bank
    dut.feature_prefetcher_req.in_features.value    = input_block_size[1]      # number of input features
    dut.feature_prefetcher_req.out_features.value   = input_block_size[0]      # number of output features
    dut.feature_prefetcher_req.nodeslot.value       = 0                         # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = precision             # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0                        # not used for weight bank requests
    number_of_clock = 1
    await RisingEdge(dut.clk)
    if blocking:
        p = 0
        while True:
            await ReadOnly()
            number_of_clock += 1
            await FallingEdge(dut.clk)
            if dut.feature_prefetcher_resp_valid.value == 1:
                logging.info("Feature prefetcher response is valid")
                break
            elif p==timeout:
                raise ValueError("Deadlock detected: feature_prefetcher_req_ready are not ready")
            p+=1
    await RisingEdge(dut.clk)
    await clear_feature_prefetcher(dut)
    logging.info("Feature prefetcher is reset")
    return number_of_clock
        
async def calculate_linear_and_writeback_b(dut, writeback_address=0x200000000, output_matrix_size=(8, 8), offset=0, precision=1, activation_code=0, bias=0, blocking=True, timeout=10000):
    dut.nsb_fte_req_valid.value = 1                                             # enable the fte
    dut.nsb_fte_req.precision.value = precision                                 # 01 is for fixed 8-bit precision
    dut.layer_config_out_channel_count.value = output_matrix_size[0]            # here we used the first dimension of the input matrix as output channel count
    dut.layer_config_out_features_count.value = output_matrix_size[1]           # here we used the first dimension of the weight matrix as output features count       
    dut.layer_config_out_features_address_msb_value.value = (writeback_address >> 32) & 0b11        # 2 is for the msb of 34 bits address
    dut.layer_config_out_features_address_lsb_value.value = writeback_address & 0xFFFFFFFF          # 0 for the rest of the address
    dut.writeback_offset.value = offset                                         # 0 for the writeback offset
    dut.layer_config_activation_function_value.value = activation_code
    if bias is not None:
        for i in range(len(bias)):
            dut.layer_config_bias_value[i].value = bias[i]
    else:
        dut.layer_config_bias_value.value = 0
    # dut.layer_config_bias_value.value = bias    
    number_of_clock = 1                                
    await RisingEdge(dut.clk)    
    if blocking:
        # while True:
        #     number_of_clock += 1
        #     await FallingEdge(dut.clk)
        #     await ReadOnly()
        #     if dut.nsb_fte_req_valid.value == 1:
        #         logging.info("FTE response is valid")
        #         break
        # await RisingEdge(dut.clk)
        # dut.nsb_fte_req_valid.value = 0
        
        p = 0
        while True:
            number_of_clock += 1
            await ReadOnly()
            await FallingEdge(dut.clk)
            if dut.nsb_fte_resp_valid.value == 1:
                logging.info("FTE response is valid")
                break
            elif p==timeout:
                raise ValueError("Deadlock detected: nsb_fte_req_ready are not ready")
            p+=1
    await RisingEdge(dut.clk)
    await clear_fte(dut)
    logging.info("FTE is reset")
    return number_of_clock