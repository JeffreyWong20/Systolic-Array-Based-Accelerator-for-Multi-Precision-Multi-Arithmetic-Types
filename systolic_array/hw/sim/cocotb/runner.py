import cocotb
import sys, os
import logging

COCOTB_LOG_LEVEL = "DEBUG"
debug = True
logger = logging.getLogger("tb_signals")
if debug:
    logger.setLevel(logging.DEBUG)

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
    )
)
print(sys.path)
from tb.runners.top_bias_tb import bias_test
from tb.runners.fcn_tb import fcn_test
# from tb.runners.top_tb import mlp_test, run_test
from systolic_array.tb.runners.mixed_precision_tb import mixed_precision_test # single mixed precision layer test
from systolic_array.tb.runners.mixed_precision_net_tb import mixed_precision_net_test # whole mixed precision net
from systolic_array.tb.runners.mase_integration import mase_mixed_precision # whole mixed precision net with weight ported form mase

@cocotb.test()
async def graph_test(dut):
    await mase_mixed_precision(dut)