library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity sequencer is
	port ( clk, reset, hold: IN STD_LOGIC;
			 output : OUT STD_LOGIC_VECTOR (9 downto 0)); -- 10 bit number will be created. 5 colors in total.
end sequencer;

architecture Behavioral of sequencer is

signal newVector,curVector,selected: std_logic_vector (9 downto 0) := "0000000000";
constant key: std_logic_vector (9 downto 0) := "0010110101";
--signal clkReduced : std_logic;
-- these signals will be used to hold and replace states as needed.


--  Characteristic Polynomial (decided randomly): P(x) = x^7 + x^3 + x + 1
--  defines XOR positions for the linear feedback. determines next output (1 bit - shift)
--  initial key (decided randomly) : 0010110101 (10 bits)


-- describe additional modules
--	COMPONENT seqClkInterface
--	PORT(
--		clkIntIn : IN std_logic;          
--		clkIntOut : OUT std_logic
--		);
--	END COMPONENT;
	
begin

	generateLFSR: process (clk, reset)
		begin
			if rising_edge (clk) then
				if reset = '1' then -- do a reset if required. then the initial key is logged in.
					curVector <= key;
				else 
					curVector <= newVector;
				end if;
			end if;
	end process;

	update: process(curVector) -- use polynomial xor statement, update first bit accordingly. shift other entries.
		begin
				newVector(9 downto 1) <= curVector (8 downto 0); -- copy shift elements to new vector.
				newVector(0) <= ((((curVector (0) XOR curVector (1)) XOR curVector (3)) XOR curVector (7)));		
				-- use xor statement given in polynomial.
	end process;-- set to output

	selectOutput: process( clk, hold, curVector)
		begin
			if rising_edge(clk) then
			if hold = '1' then
				selected <= curVector;
	--		else
	--			selected <= "0000000000"; -- Latches the value, required.
			end if;
			end if;
		end process;
	output <= selected;
--
--	Inst_seqClkInterface: seqClkInterface PORT MAP(
--		clkIntIn => clk,
--		clkIntOut => clkReduced);

end Behavioral;

