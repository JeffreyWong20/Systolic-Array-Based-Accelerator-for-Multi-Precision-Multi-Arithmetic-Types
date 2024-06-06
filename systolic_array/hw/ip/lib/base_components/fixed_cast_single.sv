`timescale 1ns / 1ps
module fixed_cast_single #(
    parameter IN_WIDTH = 8,
    parameter IN_FRAC_WIDTH = 4,
    parameter OUT_WIDTH = 4,
    parameter OUT_FRAC_WIDTH = 3
) (
    input  logic [ IN_WIDTH-1:0] data_in,
    output logic [OUT_WIDTH-1:0] data_out
);
  // TODO: Negative frac_width is not supported

  localparam IN_INT_WIDTH = IN_WIDTH - IN_FRAC_WIDTH;
  localparam OUT_INT_WIDTH = OUT_WIDTH - OUT_FRAC_WIDTH;

  // Sign
  logic out_sign;
  assign out_sign = data_in[IN_WIDTH-1];

  // Integer part
  logic [OUT_INT_WIDTH-2:0] out_int;
  if (OUT_INT_WIDTH < 2)
    assign out_int = 0;
  else 
  if (IN_INT_WIDTH > OUT_INT_WIDTH)
    assign out_int = data_in[OUT_INT_WIDTH-2+IN_FRAC_WIDTH:IN_FRAC_WIDTH];
  else
    assign out_int = {
      {(OUT_INT_WIDTH - IN_INT_WIDTH) {data_in[IN_WIDTH-1]}},
      data_in[IN_WIDTH-2:IN_FRAC_WIDTH]
    };

  // Fraction part
  logic round_up;
  logic [OUT_FRAC_WIDTH-1:0] out_frac;
  logic [OUT_WIDTH-1:0] partial_result;
  logic round_up_in_out_equal_zero, round_up_in_out_equal_zero_1;
  assign round_up_in_out_equal_zero = {out_int,out_frac}=='0 || {out_int,out_frac}=='1;
  assign round_up_in_out_equal_zero_1 = {out_int,out_frac};
  if (IN_FRAC_WIDTH > OUT_FRAC_WIDTH) begin
    assign out_frac = data_in[IN_FRAC_WIDTH-1:IN_FRAC_WIDTH-OUT_FRAC_WIDTH];
    assign partial_result = (OUT_INT_WIDTH < 2)? {out_sign,out_frac} : {out_sign,out_int,out_frac};
    /* 
    round(-0.4) = 0
    round(-0.5) = 0
    round(-0.6) = -1

    round(0.4) = 0
    round(0.5) = 0
    round(0.6) = 1

    round(1.4) = 1
    round(1.5) = 2
    round(1.6) = 2
    */

    // For positive number, if the ignored fraction bit is bigger than 0.5, round the number up (closer to 0) by +1
    // if the ignored fraction bit is =< 0.5 (correspond to fractional part >0.5), round up by +1

    // For zero integer part, round up if the ignored fraction bit is bigger than 0.5

    if (IN_FRAC_WIDTH - OUT_FRAC_WIDTH > 1) begin
      // Ignored fraction can be greater than 0.5
      always_comb begin
        if (partial_result=='0 || partial_result=='1) begin
          /*  result between -1 and 1 
              For negetive number, if the ignored fraction bit is >= 0.5, round the number up (closer to 0) by +1
              For positive number, if the ignored fraction bit is > 0.5,  round the number up (closer to 0) by +1
           */
          round_up = (data_in[IN_WIDTH-1])? data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1] : data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1] & (|data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-2:0]);
        end else begin
          /*
            For negetive/positive number, if the ignored fraction bit is > 0.5 or (=0.5 and the integer is even), round the number up by +1.
          */
          round_up = (data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1] & (|data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-2:0]));
          round_up = round_up | ((data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1] & data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH]));
        end
      end
      
    end else begin
      // Ignored fraction can only be 0.5 or 0
      always_comb begin
          if (partial_result=='0 || partial_result=='1) begin
            /*  Result between -1 and 1 
                For negetive number, if the ignored fraction bit is = 0.5, round the number up (closer to 0) by +1
                For positive number, ignored fraction bit
            */
            round_up = data_in[IN_WIDTH-1] && data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1];
          end else begin 
            /*
                For negetive/positive number, if the ignored fraction bit is = 0.5 and the integer is even, round the number up to the negative integer by +1. e.g -1.5 = -2, -2.5 = -2
            */
            round_up = data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH-1] & data_in[IN_FRAC_WIDTH-OUT_FRAC_WIDTH];
          end
      end
    end
      /* verilator lint_off WIDTH */
  end else begin
    assign out_frac = data_in[IN_FRAC_WIDTH-1:0] << (OUT_FRAC_WIDTH - IN_FRAC_WIDTH);
    assign round_up = 0;
  /* verilator lint_on WIDTH */
  end


  // Round up saturation check. TODO: This adder can be optimised out.
  logic [IN_WIDTH-1:0] rounded_up_data_in;
  if (IN_FRAC_WIDTH > OUT_FRAC_WIDTH)
    always_comb begin
      if (round_up)
        rounded_up_data_in = data_in + (round_up << IN_FRAC_WIDTH-OUT_FRAC_WIDTH);
      else
        rounded_up_data_in = data_in;
    end
  else
    assign rounded_up_data_in = data_in;
 
  logic saturate0, saturate1;
  assign saturate1 = |({(IN_WIDTH-OUT_INT_WIDTH-IN_FRAC_WIDTH){rounded_up_data_in[IN_WIDTH-1]}} ^ rounded_up_data_in[IN_WIDTH-2:OUT_INT_WIDTH-1+IN_FRAC_WIDTH]);



  // ========================================================================================================================
  // Reformatted output
  // ========================================================================================================================

  // Sign
  logic out_sign_r;
  assign out_sign_r = rounded_up_data_in[IN_WIDTH-1];

  // Integer part
  logic [OUT_INT_WIDTH-2:0] out_int_r;
  if (OUT_INT_WIDTH < 2)
    assign out_int_r = 0;
  else 
  if (IN_INT_WIDTH > OUT_INT_WIDTH)
    assign out_int_r = rounded_up_data_in[OUT_INT_WIDTH-2+IN_FRAC_WIDTH:IN_FRAC_WIDTH];
  else
    assign out_int_r = {
      {(OUT_INT_WIDTH - IN_INT_WIDTH) {rounded_up_data_in[IN_WIDTH-1]}},
      rounded_up_data_in[IN_WIDTH-2:IN_FRAC_WIDTH]
    };

  // Fraction part
  logic [OUT_FRAC_WIDTH-1:0] out_frac_r;
  if (IN_FRAC_WIDTH > OUT_FRAC_WIDTH) begin
    assign out_frac_r = rounded_up_data_in[IN_FRAC_WIDTH-1:IN_FRAC_WIDTH-OUT_FRAC_WIDTH];
  end else begin
    assign out_frac_r = rounded_up_data_in[IN_FRAC_WIDTH-1:0] << (OUT_FRAC_WIDTH - IN_FRAC_WIDTH);
  end


  if (IN_INT_WIDTH > OUT_INT_WIDTH) begin
    always_comb begin
      // Saturation check
      if (OUT_INT_WIDTH < 2) begin
          if (|({(IN_WIDTH-OUT_INT_WIDTH-IN_FRAC_WIDTH){rounded_up_data_in[IN_WIDTH-1]}} ^ rounded_up_data_in[IN_WIDTH-2:OUT_INT_WIDTH-1+IN_FRAC_WIDTH])) begin
            /* saturate to b'100...000 or b' 011..111*/
            data_out = {out_sign_r, {(OUT_WIDTH - 1) {~rounded_up_data_in[IN_WIDTH-1]}}};
          end else begin
            data_out = {out_sign_r, out_frac_r};
          end
      end else begin
          if (|({(IN_WIDTH-OUT_INT_WIDTH-IN_FRAC_WIDTH){rounded_up_data_in[IN_WIDTH-1]}} ^ rounded_up_data_in[IN_WIDTH-2:OUT_INT_WIDTH-1+IN_FRAC_WIDTH])) begin
            /* saturate to b'100...000 or b' 011..111*/
            data_out = {out_sign_r, {(OUT_WIDTH - 1) {~rounded_up_data_in[IN_WIDTH-1]}}};
          end else begin
            data_out = {out_sign_r, out_int_r, out_frac_r};
          end
      end
    end
  end else begin
    /* TODO: Having tested this part */
    if (OUT_INT_WIDTH < 2)
      assign data_out = {out_sign_r, out_frac_r};
    else
      assign data_out = {out_sign_r, out_int_r, out_frac_r};
  end


endmodule

