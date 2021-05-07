--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   20:10:26 12/15/2015
-- Design Name:   
-- Module Name:   C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/seqClkTest.vhd
-- Project Name:  SimonSaysV8
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: seqClkInterface
-- 
-- Dependencies:
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
--
-- Notes: 
-- This testbench has been automatically generated using types std_logic and
-- std_logic_vector for the ports of the unit under test.  Xilinx recommends
-- that these types always be used for the top-level I/O of a design in order
-- to guarantee that the testbench will bind correctly to the post-implementation 
-- simulation model.
--------------------------------------------------------------------------------
LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
 
-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--USE ieee.numeric_std.ALL;
 
ENTITY seqClkTest IS
END seqClkTest;
 
ARCHITECTURE behavior OF seqClkTest IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT seqClkInterface
    PORT(
         clkIntIn : IN  std_logic;
         clkIntOut : OUT  std_logic
        );
    END COMPONENT;
    

   --Inputs
   signal clkIntIn : std_logic := '0';

 	--Outputs
   signal clkIntOut : std_logic;

   -- Clock period definitions
   constant clkIntIn_period : time := 10 ns;
   constant clkIntOut_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: seqClkInterface PORT MAP (
          clkIntIn => clkIntIn,
          clkIntOut => clkIntOut
        );

   -- Clock process definitions
   clkIntIn_process :process
   begin
		clkIntIn <= '0';
		wait for clkIntIn_period/2;
		clkIntIn <= '1';
		wait for clkIntIn_period/2;
   end process;
 
   clkIntOut_process :process
   begin
		clkIntOut <= '0';
		wait for clkIntOut_period/2;
		clkIntOut <= '1';
		wait for clkIntOut_period/2;
   end process;
 

   -- Stimulus process
   stim_proc: process
   begin		
      -- hold reset state for 100 ns.
      wait for 100 ns;	

      wait for clkIntIn_period*10;

      -- insert stimulus here 

      wait;
   end process;

END;
