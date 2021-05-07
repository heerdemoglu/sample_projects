--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   22:51:33 12/17/2015
-- Design Name:   
-- Module Name:   C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/SequenceCombinerTest.vhd
-- Project Name:  SimonSaysV8
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: UserSequenceCombiner
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
 
ENTITY SequenceCombinerTest IS
END SequenceCombinerTest;
 
ARCHITECTURE behavior OF SequenceCombinerTest IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT UserSequenceCombiner
    PORT(
         clkIn : IN  std_logic;
         button : IN  std_logic_vector(3 downto 0);
         seqPartInput : IN  std_logic_vector(1 downto 0);
         softReset : IN  std_logic;
         compareEnabler : OUT  std_logic;
         userSequenceOut : OUT  std_logic_vector(9 downto 0)
        );
    END COMPONENT;
    

   --Inputs
   signal clkIn : std_logic := '0';
   signal button : std_logic_vector(3 downto 0) := (others => '0');
   signal seqPartInput : std_logic_vector(1 downto 0) := (others => '0');
   signal softReset : std_logic := '0';

 	--Outputs
   signal compareEnabler : std_logic;
   signal userSequenceOut : std_logic_vector(9 downto 0);

   -- Clock period definitions
   constant clkIn_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: UserSequenceCombiner PORT MAP (
          clkIn => clkIn,
          button => button,
          seqPartInput => seqPartInput,
          softReset => softReset,
          compareEnabler => compareEnabler,
          userSequenceOut => userSequenceOut
        );

   -- Clock process definitions
   clkIn_process :process
   begin
		clkIn <= '0';
		wait for clkIn_period/2;
		clkIn <= '1';
		wait for clkIn_period/2;
   end process;
 

   -- Stimulus process
   stim_proc: process
   begin		
      -- hold reset state for 100 ns.
      wait for 100 ns;	

      wait for clkIn_period*10;

      -- insert stimulus here 
		softReset <= '1';
		wait for 100 ns;
		softReset <= '0';
		wait for 100 ns;
		seqPartInput <= "00";
		wait for 100 ns;
		seqPartInput <= "00";
		wait for 100 ns;
		seqPartInput <= "10";
		wait for 100 ns;
		seqPartInput <= "11";
		wait for 100 ns;
		seqPartInput <= "01"; -- must return comparator enable and "0000101101" here.
		wait for 100 ns;
		seqPartInput <= "01";
		wait for 100 ns;
		seqPartInput <= "10";
		wait for 100 ns;
		seqPartInput <= "00";
		wait for 100 ns;
      wait;
   end process;

END;
