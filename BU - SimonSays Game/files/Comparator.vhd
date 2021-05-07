library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity Comparator is
	port ( comparatorClk : in std_logic;
			 CE				: in std_logic; -- compare enable
			 generatedSeq	: in std_logic_vector(9 downto 0);
			 userSeq			: in std_logic_vector(9 downto 0);
			 compared		: out std_logic); -- comparison result.
end Comparator;

architecture Behavioral of Comparator is

begin
	-- in this version compare enable is added. This resolves the issue where when 
	-- the game switches from the display stage, comparator assumes that the input and prompt
	-- are not equal, hence throwing the game to game over state.
	comparison : process ( comparatorClk, CE, generatedSeq, userSeq) begin
		if CE = '1' then
			if rising_edge(comparatorClk) then -- old design used too much different clocks.
														  -- V8 fixes the clock problems, rated clocks
				if generatedSeq = userSeq then  -- are formed within componets. (clk dividers)
					compared <= '1';
				else 
					compared <= '0';
				end if;
			end if; -- the comparison is latched and tied to CE. In early version the game assumed
		end if;    -- no input as wrong input. By using CE I can get around this problem.
	end process;


end Behavioral;

