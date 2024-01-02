import cocotb

# from tb.runners.graph_test_runner import graph_test_runner
import sys, os

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
    )
)

print(sys.path)

# from tb.runners.top_tb import simple_ram_test, 
# from tb.runners.top_tb import test_ram_operations
from tb.runners.top_tb import mlp_test
@cocotb.test()
async def graph_test(dut):
    await mlp_test(dut)