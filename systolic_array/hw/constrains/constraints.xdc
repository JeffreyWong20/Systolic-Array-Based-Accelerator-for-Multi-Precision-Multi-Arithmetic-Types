# create_clock -period 20.000 -name regbank_clk -waveform {0.000 10.000} [get_ports regbank_clk]
create_clock -period 5.000 -name clk -waveform {0.000 2.500} [get_ports clk]

# set_property C_CLK_INPUT_FREQ_HZ 300000000 [get_debug_cores dbg_hub]
# set_property C_ENABLE_CLK_DIVIDER false [get_debug_cores dbg_hub]
# set_property C_USER_SCAN_CHAIN 1 [get_debug_cores dbg_hub]
# connect_debug_port dbg_hub/clk [get_nets clk]
