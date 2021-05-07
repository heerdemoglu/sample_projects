--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   03:50:26 12/20/2015
-- Design Name:   
-- Module Name:   C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/SequencerTest.vhd
-- Project Name:  SimonSaysV8
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: sequencer
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
 
ENTITY SequencerTest IS
END SequencerTest;
 
ARCHITECTURE behavior OF SequencerTest IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT sequencer
    PORT(
         clk : IN  std_logic;
         reset : IN  std_logic;
         hold : IN  std_logic;
         output : OUT  std_logic_vector(9 downto 0)
        );
    END COMPONENT;
    

   --Inputs
   signal clk : std_logic := '0';
   signal reset : std_logic := '0';
   signal hold : std_logic := '0';

 	--Outputs
   signal output : std_logic_vector(9 downto 0);

   -- Clock period definitions
   constant clk_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: sequencer PORT MAP (
          clk => clk,
          reset => reset,
          hold => hold,
          output => output
        );

   -- Clock process definitions
   clk_process :process
   begin
		clk <= '0';
		wait for clk_period/2;
		clk <= '1';
		wait for clk_period/2;
   end process;
 

   -- Stimulus process
   stim_proc: process
   begin		
      -- hold reset state for 100 ns.
      wait for 100 ns;	

      wait for clk_period*10;

      -- insert stimulus here 
		reset <= '1';
		wait for 100 ns;
		reset <= '0';
		wait for 100 ns;
		hold <= '1';
		wait for 100 ns;
		hold <= '0';
		wait for 300 ns;
		hold <= '1';
		wait for 100 ns;
		hold <= '0';
		wait for 100 ns;
      wait;

      wait;
   end process;

END;
