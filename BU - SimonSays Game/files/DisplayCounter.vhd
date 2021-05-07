library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

use IEEE.NUMERIC_STD.ALL;

entity DisplayClkDiv is
	port ( clkIn : in std_logic;
			 clkDisplay : out std_logic);
end DisplayClkDiv;

architecture Behavioral of DisplayClkDiv is
-- this component is used to decrease the clock to 1 Hz clock.
-- this allows the colors to stay on screen for a brief moment of time.
-- hardness can be changed from here. Faster the clk divider is, harder for
-- players to memorize the pattern. (further implementation)

signal temp : std_logic;
constant divider : integer := 25000000; -- 1 Hz.
signal counter : integer := 0;

begin

	clkDivider : process (clkIn, temp) begin
		if rising_edge(clkIn) then
			if counter = divider then
				temp <= not temp;
				counter <= 0;
			else
				counter <= counter + 1;
			end if;
		end if;
		clkDisplay <= temp;
	end process;



end Behavioral;

