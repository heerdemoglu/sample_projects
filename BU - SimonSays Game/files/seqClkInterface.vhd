library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity seqClkInterface is
	port ( clkIntIn : in std_logic;
			 clkIntOut: out std_logic);
end seqClkInterface;

architecture Behavioral of seqClkInterface is

-- signal declerations
constant divider : integer := 1000000;
signal counter: integer := 0;
signal temp : std_logic := '0';

begin

	clkDivider : process (clkIntIn, temp) begin
		if rising_edge(clkIntIn) then
			if counter = divider then
				temp <= not temp;
				counter <= 0;
			else
				counter <= counter + 1;
			end if;
		end if;
		clkIntOut <= temp;
	end process;

end Behavioral;

