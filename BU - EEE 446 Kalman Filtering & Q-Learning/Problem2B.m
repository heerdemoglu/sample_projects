% *************************************************************************
% EEE 446 - MATLAB Assignment - Problem 2 Part B:
% Hasan Emre Erdemoglu - 21401462 - Summer 2019
% References: Lecture Notes & RL by Sutton & Barto
% *************************************************************************
clear; clc; close all; % For debugging purposes

% Fix arbitrary discount factor btwn (0,1):
beta = 0.62; % Fixed for debugging purposes
eps = 0.00001; % Difference between two consec. iterations to stop VI

% Look-up table for this question is already built in:
% Probability of state transition using a particular action.
P1 = [0.1 0.9; 0.8 0.2];  P2 = [0.5 0.5; 0.2 0.8];
c = [0 0.5; 0 -0.5]; % Transition costs ([1->1 1->2 2->1 2->2])