--------------------------------------------------------------------------------
-- Company: 
-- Engineer:
--
-- Create Date:   14:46:48 12/20/2015
-- Design Name:   
-- Module Name:   C:/Users/Emre Erdemoglu/Desktop/EEE102Project/SimonSaysV8/DisplaySeqStageTest.vhd
-- Project Name:  SimonSaysV8
-- Target Device:  
-- Tool versions:  
-- Description:   
-- 
-- VHDL Test Bench Created by ISE for module: DisplaySeqStage
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
 
ENTITY DisplaySeqStageTest IS
END DisplaySeqStageTest;
 
ARCHITECTURE behavior OF DisplaySeqStageTest IS 
 
    -- Component Declaration for the Unit Under Test (UUT)
 
    COMPONENT DisplaySeqStage
    PORT(
         dataEN : IN  std_logic;
         clk : IN  std_logic;
         resetDispIn : IN  std_logic;
         resetSeq : OUT  std_logic;
         holdIn : IN  std_logic;
         outputSeq : OUT  std_logic_vector(9 downto 0);
         stageSwitch : OUT  std_logic;
         toVGADispStage : OUT  std_logic_vector(7 downto 0)
        );
    END COMPONENT;
    

   --Inputs
   signal dataEN : std_logic := '0';
   signal clk : std_logic := '0';
   signal resetDispIn : std_logic := '0';
   signal holdIn : std_logic := '0';

 	--Outputs
   signal resetSeq : std_logic;
   signal outputSeq : std_logic_vector(9 downto 0);
   signal stageSwitch : std_logic;
   signal toVGADispStage : std_logic_vector(7 downto 0);

   -- Clock period definitions
   constant clk_period : time := 10 ns;
 
BEGIN
 
	-- Instantiate the Unit Under Test (UUT)
   uut: DisplaySeqStage PORT MAP (
          dataEN => dataEN,
          clk => clk,
          resetDispIn => resetDispIn,
          resetSeq => resetSeq,
          holdIn => holdIn,
          outputSeq => outputSeq,
          stageSwitch => stageSwitch,
          toVGADispStage => toVGADispStage
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
		dataen <= '1';
		resetDispIn <= '1';
		wait for 100 ns;
		resetDispIn <= '0';
		wait for 100 ns;
		holdIn <= '1';
		wait for 100 ns;
		holdIn <= '0';
		wait for 100 ns;
		resetDispIn <= '1';
		wait for 100 ns;
		resetDispIn <= '0';
		wait for 100 ns;
		dataen <= '0';
		resetDispIn <= '1';
		wait for 100 ns;
		resetDispIn <= '0';
		wait for 100 ns;
		holdIn <= '1';
		wait for 100 ns;
		holdIn <= '0';
		wait for 100 ns;
		resetDispIn <= '1';
		wait for 100 ns;
		resetDispIn <= '0';
		wait for 100 ns;

      wait;
   end process;

END;
