//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:
// Design Name: 
// Module Name: mac
// Project Name:
// Target Devices:
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module mac #(
    parameter PRECISION   = top_pkg::FLOAT_32,
    parameter DATA_WIDTH  = 32,
    parameter ACCUMULATOR_WIDTH = 32
) (
    input  logic                              core_clk,            
    input  logic                              resetn,

    input  logic                              in_valid,
    output logic                              in_ready,

    input  logic [DATA_WIDTH-1:0]             a,
    input  logic [DATA_WIDTH-1:0]             b,

    output logic [ACCUMULATOR_WIDTH-1:0]     accumulator,
    
    input  logic                             overwrite,
    input  logic [ACCUMULATOR_WIDTH-1:0]     overwrite_data
    
);

logic float_mac_valid; // TODO: Remove this later

if (PRECISION == top_pkg::FLOAT_32) begin

    // float_mac #(
    //     .FLOAT_WIDTH (FLOAT_WIDTH)
    // ) float_mac_i (
    //     .core_clk,            
    //     .resetn,
        
    //     .in_valid,
    //     .in_ready,
        
    //     .a,
    //     .b,
        
    //     .overwrite,
    //     .overwrite_data,
        
    //     .accumulator
    // );
    assign float_mac_valid = 1;

end else begin

    fixed_point_mac #(
        .DATA_WIDTH     (DATA_WIDTH),
        .ACCUMULATOR_WIDTH (ACCUMULATOR_WIDTH)
    ) fixed_point_mac_i (
        .core_clk,            
        .resetn,
        
        .in_valid,
        .in_ready,
        
        .a,
        .b,
        
        .overwrite,
        .overwrite_data,
        
        .accumulator
    );

end

// ======================================================================================================
// Assertions
// ======================================================================================================

P_acc_constant: cover property (
    @(posedge core_clk) disable iff (!resetn)
    (in_valid && in_ready) |=> (accumulator == $past(accumulator, 1))
);

endmodule

