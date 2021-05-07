library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;

entity EndGameStage is
	port ( clkInEnd : in std_logic;
			 OverEN :   in std_logic;
			 resetOver : in std_logic;
			 buttonEN : out std_logic;
			 toVGAGameOver : out std_logic_vector ( 7 downto 0));
			 
end EndGameStage;

architecture Behavioral of EndGameStage is

-- component decleration
	COMPONENT gameOver
	PORT(
		EN : IN std_logic;
		reset : IN std_logic;
		toVGAEnd : OUT std_logic_vector(7 downto 0)
		);
	END COMPONENT;
	
-- signal init
	type state is (over, idle);
	signal curr,nxt : state;

	
begin
	
	-- component init
	
	Inst_gameOver: gameOver PORT MAP(
			EN => overEN,
			reset => resetOver,
			toVGAEnd => toVGAGameOver);
			
	-- inner processes
	
	-- state changer
	change : process (clkInEnd) begin
		if rising_edge (clkInEnd) then 
			curr <= nxt;
		end if;
	end process;
	
	-- machine interactions
	FSMOver : process ( curr, overEN)  begin
		-- enable disable button to avoid IO conflicts.
		buttonEN <= '1';
		
		case curr is
			when idle =>
				if overEN <= '1' then
					nxt <= over;
				else
					nxt <= idle;
				end if;
			when over => 
				if overEN <= '0' then
					nxt <= idle;
				else
					buttonEN <= '0';
					nxt <= over;
				end if;
			end case;
	end process;

end Behavioral;

