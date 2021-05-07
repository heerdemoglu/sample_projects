% *************************************************************************
% EEE 446 - MATLAB Assignment - Problem 2 Part A-iii:
% Hasan Emre Erdemoglu - 21401462 - Summer 2019
% References: Lecture Notes & RL by Sutton & Barto
% *************************************************************************
clear; clc; close all; % For debugging purposes

% Fix arbitrary discount factor btwn (0,1):
beta = 0.62; % Fixed for debugging purposes
eps = 0.00001; % Difference between two consec. iterations to stop VI
max_ep = 1000;
max_stp = 100;

% Look-up table for this question is already built in:
% Probability of state transition using a particular action.
P1 = [0.1 0.9; 0.8 0.2];  P2 = [0.5 0.5; 0.2 0.8];
c = [0 0.5; 0 -0.5]; % Transition costs ([1->1 1->2 2->1 2->2])

% Implement Q-Learning Algorithm:
Q = zeros(2,2); % State-Action Space is 2x2. Arbitrarily assigned.

for ep = 1:max_ep
	% Initialize x
	x = 1; % Starting state is 1, init randomly
% 	x = randi([1 2],1,1); % Normally random selected starting state
	for stp = 1:max_stp
		
	end
	
end