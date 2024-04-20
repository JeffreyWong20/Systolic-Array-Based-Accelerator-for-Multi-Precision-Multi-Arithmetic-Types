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
from tb.runners.top_tb import mlp_test, run_test

@cocotb.test()
async def graph_test(dut):
    await mlp_test(dut)