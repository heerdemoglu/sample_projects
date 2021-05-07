--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   03:21:34 12/20/2015
-- Design Name:   
-- Module Name:   C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/DisplaySeqTest.vhd
-- Project Name:  SimonSaysV8
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: DisplaySequence
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
 
ENTITY DisplaySeqTest IS
END DisplaySeqTest;
 
ARCHITECTURE behavior OF DisplaySeqTest IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT DisplaySequence
    PORT(
         start : IN  std_logic;
         resetDisp : IN  std_logic;
         clkInDisp : IN  std_logic;
         holdDisp : IN  std_logic;
         toVGADisplay : OUT  std_logic_vector(7 downto 0);
         generatedSequenceOut : OUT  std_logic_vector(9 downto 0)
        );
    END COMPONENT;
    

   --Inputs
   signal start : std_logic := '0';
   signal resetDisp : std_logic := '0';
   signal clkInDisp : std_logic := '0';
   signal holdDisp : std_logic := '0';

 	--Outputs
   signal toVGADisplay : std_logic_vector(7 downto 0);
   signal generatedSequenceOut : std_logic_vector(9 downto 0);

   -- Clock period definitions
   constant clkInDisp_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: DisplaySequence PORT MAP (
          start => start,
          resetDisp => resetDisp,
          clkInDisp => clkInDisp,
          holdDisp => holdDisp,
          toVGADisplay => toVGADisplay,
          generatedSequenceOut => generatedSequenceOut
        );

   -- Clock process definitions
   clkInDisp_process :process
   begin
		clkInDisp <= '0';
		wait for clkInDisp_period/2;
		clkInDisp <= '1';
		wait for clkInDisp_period/2;
   end process;
 

   -- Stimulus process
   stim_proc: process
   begin		
      -- hold reset state for 100 ns.
      wait for 100 ns;	

      wait for clkInDisp_period*10;

      -- insert stimulus here 
		resetDisp <= '1';
		wait for 100 ns;
		resetDisp <= '0';
	
		holdDisp <= '1';
		wait for 100 ns;
		
		holdDisp <= '0';
		start <= '1';
		wait for 100 ns;
		start <= '0';

      wait;
   end process;

END;
