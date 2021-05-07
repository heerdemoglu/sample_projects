% *************************************************************************
% EEE 446 - MATLAB Assignment - Problem 2 Part A-ii:
% Hasan Emre Erdemoglu - 21401462 - Summer 2019
% References: Lecture Notes & RL by Sutton & Barto
% This is almost the same with Policy Evaluation in the previous section.
% Here we do not update by iteratively improving policy but the action.
% *************************************************************************
clear; clc; close all; % For debugging purposes

% Fix arbitrary discount factor btwn (0,1):
beta = 0.62; % Fixed for debugging purposes
eps = 0.00001; % Difference between two consec. iterations to stop VI

% Look-up table for this question is already built in:
% Probability of state transition using a particular action.
P1 = [0.1 0.9; 0.8 0.2];  P2 = [0.5 0.5; 0.2 0.8];
c = [0 0.5; 0 -0.5]; % Transition costs ([1->1 1->2 2->1 2->2])

% Implement Value Iteration Algorithm:
V = [0; 0]; % Assign initial value.
max_it = 1000; % Max iter. keeps loop terminate at some point

for it = 1:max_it
	Vtemp = V;
	delta = 0;
	
	% Keep the function in matrix form so that we can find the minimum of
	% elements (1 or 2):
	% Update rule: V(s) <- (P_ss')^a * ((R_ss')^a)+beta*V(s'))
	% Take the minimum of either a (or u) = 1 or a = 2.
	
	% Do it for state 1:
	[V1(it), pol1] = min( ... 
		[sum(P1(1,:)'.*(c(:,1)+beta*Vtemp)), ... 
		 sum(P2(1,:)'.*(c(:,2)+beta*Vtemp))]);
	
	% Do it for state 2:
	[V2(it), pol2] = min( ... 
		[sum(P1(2,:)'.*(c(:,1)+beta*Vtemp)), ... 
		 sum(P2(2,:)'.*(c(:,2)+beta*Vtemp))]);
	 	
	% Check if converged:
	delta = max(delta,max(abs(V(1)-V1(it)),abs(V(2)-V2(it))));
	V = [V1(it); V2(it)]; 
	pol(:,it) = [pol1; pol2]; clear pol1; clear pol2;
	
	% Break out of the loop if convergence condition is reached
	if (delta < eps)
		break;
	end
end

% Print out the results:
figure; title('Value Functions with respect to Iterations');
xlabel('Iterations'); ylabel('Value');
hold on; axis tight; grid on;
plot(V1); plot(V2); legend('V1', 'V2');

disp('Policies selected (First Row - State 1, Second Row - State 2):')
disp(pol)
fprintf('Convergence occurred after %d iterations. \n', it);


