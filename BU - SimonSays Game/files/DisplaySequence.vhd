library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;

entity DisplaySequence is
	port ( start : in std_logic;
			 resetDisp : in std_logic;
			 clkInDisp : in std_logic;
			 holdDisp : in std_logic;
			 toVGADisplay : out std_logic_vector(7 downto 0);
			 generatedSequenceOut : out std_logic_vector(9 downto 0);
			 counter : out std_logic_vector(2 downto 0));
end DisplaySequence;

architecture Behavioral of DisplaySequence is

	COMPONENT sequencer
		PORT(
			clk : IN std_logic;
			reset : IN std_logic;
			hold : IN std_logic;          
			output : OUT std_logic_vector(9 downto 0)
			);
		END COMPONENT;	
		
	COMPONENT DisplayClkDiv
		PORT(
			clkIn : IN std_logic;          
			clkDisplay : OUT std_logic
			);
		END COMPONENT;

	COMPONENT ShiftRegister
		PORT(
			input : IN std_logic;
			regclock : IN std_logic;          
			output : OUT std_logic_vector(1 downto 0)
			);
		END COMPONENT;


	signal clkUsed : std_logic;
	signal genSeq : std_logic_vector(9 downto 0);
	signal temp: std_logic_vector(1 downto 0);
	signal cout : std_logic_vector(2 downto 0) := "000";
	
	begin
	
		generatedSequenceOut <= genSeq;
		counter <= cout;

		Inst_DisplayClkDiv: DisplayClkDiv PORT MAP(
			clkIn => clkInDisp,
			clkDisplay => clkUsed);

		Inst_sequencer: sequencer PORT MAP(
			clk => clkInDisp,
			reset => resetDisp,
			hold => holdDisp,
			output =>  genSeq);
		
		Inst_ShiftRegister: ShiftRegister PORT MAP(
			input => genSeq(0),
			regclock => clkInDisp,
			output => temp);
		
	DisplaySequence : process (start,resetDisp, temp, clkUsed) begin
		if rising_edge(clkUsed) then -- 1 Hz cannot be shown in test bench, convert back to 50 mhz to see testbench.
			if resetDisp = '1' then
				toVGADisplay <= "00000000";
			elsif start = '1' then
				case temp is
					when "00" => toVGADisplay <= "11100000"; --red
					when "01" => toVGADisplay <= "00011100"; --green
					when "10" => toVGADisplay <= "00000011"; -- blue
					when others => toVGADisplay <= "11111100"; -- yellow
				end case;		
			end if;
		end if;
	end process;
	
	counterProcess : process ( cout, clkUsed, start, resetDisp ) begin
		if resetDisp = '1' then
			cout <= "000";
		elsif (start = '1' and ((cout /= "101" or cout /= "110") or cout /= "111")) then
			if rising_edge(clkUsed) then
				cout <= cout + 1;
			end if;
		end if;
	end process;
	
end Behavioral;

