import cocotb
from cocotb.triggers import RisingEdge
import math

class AXIDriver:
    def __init__(self, dut):
        self.dut = dut

    # This function writes data to the AXI interface
    
    async def axi_write(self, address, data):
        # Handle unaligned address   
        wstrd = 0xFFFFFFFFFFFFFFFF
        address_valid = address >> 6
        address_valid = address_valid << 6
        address_diff = address - address_valid
        data = data << (address_diff * 8)
        
        if (address_diff != 0): 
            wstrd = wstrd << address_diff
            wstrd = int(hex(wstrd)[address_diff:], 16)
            
        # Reset signals    
        self.dut.s_axi_awaddr.value     = 0
        self.dut.s_axi_awburst.value    = 0
        self.dut.s_axi_awcache.value    = 0
        self.dut.s_axi_awid.value       = 0
        self.dut.s_axi_awlen.value      = 0
        self.dut.s_axi_awlock.value     = 0
        self.dut.s_axi_awprot.value     = 0
        self.dut.s_axi_awsize.value     = 0
        self.dut.s_axi_awvalid.value    = 0
        
        
        self.dut.s_axi_wdata.value  = 0
        self.dut.s_axi_wlast.value  = 0
        self.dut.s_axi_wstrb.value  = wstrd
        self.dut.s_axi_wvalid.value = 0

        self.dut.s_axi_bready.value = 0

        await RisingEdge(self.dut.clk)

        # AW phase
        self.dut.s_axi_awvalid.value    = 1
        self.dut.s_axi_awaddr.value     = address
        self.dut.s_axi_awsize.value     = 2 # Set writing 32 bits in one go 2^2 = 4 bytes
        self.dut.s_axi_awburst.value    = 0 # fixed address
        self.dut.s_axi_awlen.value      = 0 # Single beat transaction
        # Wait to accept address
        print("Waiting for awready")
        while(True):
            await RisingEdge(self.dut.clk)
            if (self.dut.s_axi_awready.value):
                break
        print("awready received")

        self.dut.s_axi_awvalid.value = 0

        # W phase (single beat transaction)
        self.dut.s_axi_wvalid.value     = 1
        self.dut.s_axi_wdata.value      = data
        # Wait to accept data
        while(True):
            await RisingEdge(self.dut.clk)
            if (self.dut.s_axi_wready.value):
                break
        print("wready received")
        self.dut.s_axi_wvalid.value = 0
        self.dut.s_axi_bready.value = 1
        while(True):
            await RisingEdge(self.dut.clk)
            if (self.dut.s_axi_bvalid.value):
                break
        print("bvalid received")
        self.dut.s_axi_bready.value = 0

    async def axi_read(self, address):
        await RisingEdge(self.dut.clk)
        self.dut.s_axi_araddr.value     = 0
        self.dut.s_axi_arburst.value    = 0
        self.dut.s_axi_arcache.value    = 0
        self.dut.s_axi_arid.value       = 0
        self.dut.s_axi_arlen.value      = 0 # Single beat transaction
        self.dut.s_axi_arlock.value     = 0
        self.dut.s_axi_arprot.value     = 0
        self.dut.s_axi_arsize.value     = 0 # Single byte per transaction
        self.dut.s_axi_arvalid.value    = 0
        
        self.dut.s_axi_rready.value     = 0

        await RisingEdge(self.dut.clk)
        print("Waiting for arready at address: ", hex(address))
        # AR phase
        self.dut.s_axi_arvalid.value    = 1
        self.dut.s_axi_araddr.value     = address

        # Wait to accept address
        while(True):
            await RisingEdge(self.dut.clk)
            if (self.dut.s_axi_arready.value):
                self.dut.s_axi_arvalid.value    = 0
                break
        print("arready received")

        self.dut.s_axi_rready.value     = 1

        # Wait to accept data
        while(True):
            await RisingEdge(self.dut.clk)
            if (self.dut.s_axi_rvalid.value):
                self.dut.s_axi_rready.value = 0
                print("rdata: ", self.dut.s_axi_rdata.value.hex())
                print(self.dut.s_axi_rdata.value)
                return self.dut.s_axi_rdata.value

    async def reset_axi_interface(self):
        self.dut.s_axi_awvalid.value = 0
        self.dut.s_axi_awaddr.value = 0
        self.dut.s_axi_awprot.value = 0
        self.dut.s_axi_wvalid.value = 0
        self.dut.s_axi_wdata.value = 0
        self.dut.s_axi_wstrb.value = 0
        self.dut.s_axi_bready.value = 0
        self.dut.s_axi_arvalid.value = 0
        self.dut.s_axi_araddr.value = 0
        self.dut.s_axi_arprot.value = 0
        self.dut.s_axi_rready.value = 0
