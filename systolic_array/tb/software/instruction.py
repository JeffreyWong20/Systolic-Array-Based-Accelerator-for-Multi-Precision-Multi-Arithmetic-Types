# --------------------------------------------------
# Instruction
# --------------------------------------------------
def load_weight_instruction(dut, start_address=0x0000, weight_matrix_size=(4, 128), precision=1):
    dut.weight_prefetcher_req_valid.value           = 1                         # enable the prefetcher
    dut.weight_prefetcher_req.req_opcode.value      = 0                         # 00 is for weight bank requests
    dut.weight_prefetcher_req.start_address.value   = start_address             # start address of the weight bank
    dut.weight_prefetcher_req.in_features.value     = weight_matrix_size[1]     # number of input features                     
    dut.weight_prefetcher_req.out_features.value    = weight_matrix_size[0]     # number of output features
    dut.weight_prefetcher_req.nodeslot.value        = 0                         # not used for weight bank requests
    dut.weight_prefetcher_req.nodeslot_precision.value = precision              # 01 is for fixed 8-bit precision
    dut.weight_prefetcher_req.neighbour_count.value = 0                         # not used for weight bank requests

def load_feature_instruction(dut, start_address=0x0000, input_matrix_size=(8, 128), precision=1):
    dut.feature_prefetcher_req_valid.value          = 1                         # enable the prefetcher
    dut.feature_prefetcher_req.req_opcode.value     = 0                         # 00 is for weight bank requests
    dut.feature_prefetcher_req.start_address.value  = start_address             # start address of the feature bank
    dut.feature_prefetcher_req.in_features.value    = input_matrix_size[1]      # number of input features
    dut.feature_prefetcher_req.out_features.value   = input_matrix_size[0]      # number of output features
    dut.feature_prefetcher_req.nodeslot.value       = 0                         # not used for weight bank requests
    dut.feature_prefetcher_req.nodeslot_precision.value = precision             # 01 is for fixed 8-bit precision
    dut.feature_prefetcher_req.neighbour_count.value = 0                        # not used for weight bank requests