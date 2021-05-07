% *************************************************************************
% EEE 446 - MATLAB Assignment - Problem 1
% Hasan Emre Erdemoglu - 21401462 - Summer 2019
% References: Lecture Notes
% *************************************************************************
clear; clc; close all; % For debugging purposes

% Given entities:
A = [1 1; 0 1]; C = [2 1]; w = eye(2); v = 1; s0 = eye(2);
% Start from initial cond. Recurse until consecutive steps are eps away
% from each other:
s_next = s0; eps = 0.00001;

max_it = 1000;% If not converged, stop after this limit.
for k=1:max_it
	% Keep current state in temp, calculate next recursion, look at
	% absolute difference:
	s_cur = s_next;
	
	s_next = A*s_next*A'+w-(A*s_next*C')/(C*s_next*C'+v)*(C*s_next*A');
	
	
	if abs(s_cur - s_next) <= eps
		fprintf('Convergence achieved in %d steps. \n',k);
		fprintf('Unique fixed point given as follows: \n');
		disp(s_next);
		break;
	end
end

