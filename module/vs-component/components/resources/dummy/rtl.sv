/*******************************************************************************
 * Component: dummy
 * Type: resource
 * Description: Do nothing
 ******************************************************************************/


/*******************************************************************************
 * Package
 ******************************************************************************/
package dummy_pkg;
    parameter RESOURCE_INSTR_WIDTH = 27;

    // Others:


endpackage

/*******************************************************************************
 * Module
 ******************************************************************************/
module dummy
import dummy_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    input  logic instr_en,
    input  logic [RESOURCE_INSTR_WIDTH-1:0] instr,
    input  logic activate
);

    // Parameter check:

    // Function definition:

endmodule