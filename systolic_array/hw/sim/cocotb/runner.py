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
from tb.runners.top_tb import mlp_test, run_test
# from tb.runners.jsc_tb import jsc_test

@cocotb.test()
async def graph_test(dut):
    await bias_test(dut)