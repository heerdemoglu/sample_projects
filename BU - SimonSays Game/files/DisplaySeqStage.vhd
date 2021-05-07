library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;

entity DisplaySeqStage is
	port ( dataEN : in std_logic;
			 clk	  : in std_logic;
			 resetDispIn  : in std_logic;
			 resetSeq 	: out std_logic;
			 holdIn		: in std_logic;
			 outputSeq  : out std_logic_vector(9 downto 0);
			 stageSwitch : out std_logic;
			 toVGADispStage : out std_logic_vector(7 downto 0));
end DisplaySeqStage;

architecture Behavioral of DisplaySeqStage is

	COMPONENT DisplaySequence
		PORT(
			start : IN std_logic;
			resetDisp : IN std_logic;
			clkInDisp : IN std_logic;
			holdDisp : IN std_logic;          
			toVGADisplay : OUT std_logic_vector(7 downto 0);
			generatedSequenceOut : OUT std_logic_vector(9 downto 0);
			counter : OUT std_logic_vector(2 downto 0)
			);
	END COMPONENT;


	type state is (active, idle);
	signal curr, nxt : state;
	signal count : std_logic_vector(2 downto 0);
	
begin

	Inst_DisplaySequence: DisplaySequence PORT MAP(
		start => dataEN,
		resetDisp => resetDispIn,
		clkInDisp => clk,
		holdDisp => holdIn,
		toVGADisplay => toVGADispStage,
		generatedSequenceOut => outputSeq,
		counter => count	);

	-- state switch
	FSMInit : process ( clk, dataEN, resetDispIn) begin
		if resetDispIn = '1' or dataEN = '0' then
			curr <= idle;
		elsif rising_edge(clk) then
			curr <= nxt;
		end if;
	end process;
	
	--FSM description
	FSMDisplayStage : process ( curr, dataEN, count ) begin
		resetSeq <= '0';
		stageSwitch <= '0';
		
		case curr is 
			when active => 
				if count = "100" then
					nxt <= idle;
					stageSwitch <= '1';
				else
					nxt <= active;
				end if;
			when idle => 
				if dataEN = '1' then
					nxt <= active;
				elsif dataEN = '0' then
					resetSeq <= '1';
					nxt <= idle;
				else
					null; -- a good function to not to create latches out of nowhere. From a VHDL book.
				end if;
		end case;
	end process;

end Behavioral;

