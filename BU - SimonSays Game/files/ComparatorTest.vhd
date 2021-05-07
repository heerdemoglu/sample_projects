--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   22:04:45 12/15/2015
-- Design Name:   
-- Module Name:   C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/ComparatorTest.vhd
-- Project Name:  SimonSaysV8
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: Comparator
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
 
ENTITY ComparatorTest IS
END ComparatorTest;
 
ARCHITECTURE behavior OF ComparatorTest IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT Comparator
    PORT(
         comparatorClk : IN  std_logic;
         CE : IN  std_logic;
         generatedSeq : IN  std_logic_vector(9 downto 0);
         userSeq : IN  std_logic_vector(9 downto 0);
         compared : OUT  std_logic
        );
    END COMPONENT;
    

   --Inputs
   signal comparatorClk : std_logic := '0';
   signal CE : std_logic := '0';
   signal generatedSeq : std_logic_vector(9 downto 0) := (others => '0');
   signal userSeq : std_logic_vector(9 downto 0) := (others => '0');

 	--Outputs
   signal compared : std_logic;

   -- Clock period definitions
   constant comparatorClk_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: Comparator PORT MAP (
          comparatorClk => comparatorClk,
          CE => CE,
          generatedSeq => generatedSeq,
          userSeq => userSeq,
          compared => compared
        );

   -- Clock process definitions
   comparatorClk_process :process
   begin
		comparatorClk <= '0';
		wait for comparatorClk_period/2;
		comparatorClk <= '1';
		wait for comparatorClk_period/2;
   end process;
 

   -- Stimulus process
   stim_proc: process
   begin		
      -- hold reset state for 100 ns.
      wait for 100 ns;	

      wait for comparatorClk_period*10;

      -- insert stimulus here
		CE <= '0';
		wait for 100 ns;
		generatedSeq <= "0110111010";
		wait for 100 ns;
		userSeq <= "0110111010";
		wait for 100 ns;
		userSeq <= "0110100110";
		wait for 100 ns;
		CE <= '1';
		wait for 100 ns;
		userSeq <= "0110111010";
		wait for 100 ns;
		userSeq <= "0110100110";
		wait for 100 ns;

      wait;
   end process;

END;
