library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;


entity ShiftRegister is
    Port ( input : in  STD_LOGIC; -- button input
           regclock : in  STD_LOGIC;
           output : out  STD_LOGIC_vector(1 downto 0));
end ShiftRegister;

architecture Behavioral of ShiftRegister is

signal Q : std_logic_vector (1 downto 0);

begin

	process (regclock) begin
		if rising_edge(regclock) then
			Q(0) <= input;
			Q(1) <= Q(0);			
		end if;
	end process;
	
	output <= Q(1 downto 0);		
end Behavioral;

