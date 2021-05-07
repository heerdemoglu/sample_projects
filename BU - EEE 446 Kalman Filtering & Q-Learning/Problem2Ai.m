% *************************************************************************
% EEE 446 - MATLAB Assignment - Problem 2 Part A-i:
% Hasan Emre Erdemoglu - 21401462 - Summer 2019
% References: Lecture Notes & RL by Sutton & Barto
% Note: I did Value Iteration first as I it felt easier to deal with.
% *************************************************************************
clear; clc; close all; % For debugging purposes

% Fix arbitrary discount factor btwn (0,1):
beta = 0.62; % Fixed for debugging purposes
eps = 0.00001; % Difference between two consec. iterations to stop VI

% Look-up table for this question is already built in:
% Probability of state transition using a particular action.
P1 = [0.1 0.9; 0.8 0.2];  P2 = [0.5 0.5; 0.2 0.8];
c = [0 0.5; 0 -0.5]; % Transition costs ([1->1 1->2 2->1 2->2])

% 1. Initialization:
V = [rand; rand]; % Assign initial value.

% Run the algorithm
max_it = 1000; % Max iter. keeps loop terminate at some point
% At any case less than max_it policies will be used. - Remaining garbage.
pols = randi([1 2],2,1); % init policies decided randomly
count = 1;
Vgraph(:,count) = V;
polsGraph(:,count) = pols; itpast = 1; it = 1;
while it <= max_it
	Vtemp = V;
	delta = 0;
	
	[V] = polEval(P1, P2, pols, beta, c, Vtemp)
	delta = max(delta,max((abs(V(1)-Vtemp(1))),abs(V(2)-Vtemp(2))));
	
	% for graphing over all policy improvements
	Vgg(:,itpast) = V; 
	itpast = itpast + 1;
	it = it + 1;
	
	% Break out of the loop if convergence condition is reached
	if (delta < eps)
		% for scorekeeping:
		count = count + 1;
		Vgraph(:,count) = V;
		iterationState(count-1) = it;
		
		% now continue to policy improvement
		stableflag = 1;
		cur_pol = pols; % we may truncate unnecessary policies
		% calculate new optimal policy, if stable get out
		[pol_next] = polImp(P1, P2, beta, c, V);
		
		% if not stable then it = 1, return to policy evaluation with new 
		% policy pi.
		if cur_pol(1) ~= pol_next(1) || cur_pol(2) ~= pol_next(2) % x stbl
			stableflag = 0;
			it = 1;
			pols = pol_next;
			polsGraph(:,count) = pols;
		else % stable
			break;
		end
	end
end


% Documentation of results:
fprintf('%d Policy Improvement steps were made.');
disp('After each policy improvement this many iterations were taken to reach to stable point ');
disp(iterationState);

disp('Stabilized state values at each policy evaluation:')
disp(Vgraph);

disp('At the end of the algorithm under optimal policy, we reach to optimal state-value ');
disp(V);

disp('Chosen policies at each policy improvements:')
disp(polsGraph);

% Print out the results:
figure; title('Policy Functions with respect to Iterations');
xlabel('Iterations'); ylabel('Value');
hold on; axis tight; grid on;
plot(Vgg(1,:)); plot(Vgg(2,:)); legend('V1', 'V2');

fprintf('Convergence occurred after %d iterations. \n', itpast);



% 2. Policy Evaluation & Improvement Chunklets:
% prob1, prob2 2x2 state transition array with action 1 and 2 respectively
% policy is the policy that we want to find value of
% beta is the discount factor
% cost is 2x2 cost metric
% Vprev is the initial value that we give, Vnext is the output
function [Vnext] = polEval(prob1, prob2, policy, beta, cost, Vprev)

% Do it for state 1:
if policy(1) == 1 % picked action 1
	V1 = sum(prob1(1,:)'.*(cost(:,1)+beta*Vprev));
else % picked action 2
	V1 = sum(prob2(1,:)'.*(cost(:,2)+beta*Vprev));
end

% Do it for state 2:
if policy(2) == 1 % picked action 1
	V2 = sum(prob1(2,:)'.*(cost(:,1)+beta*Vprev));
else % picked action 2
	V2 = sum(prob2(2,:)'.*(cost(:,2)+beta*Vprev));
end

Vnext = [V1; V2];

end

function [pol_next] = polImp(prob1, prob2, beta, cost, Vprev)
% Keep the function in matrix form so that we can find the minimum of
% elements (1 or 2):
% Update rule: V(s) <- (P_ss')^a * ((R_ss')^a)+beta*V(s'))
% Take the minimum of either a (or u) = 1 or a = 2.

% Do it for state 1:
[~, pol1] = min( ...
	[sum(prob1(1,:)'.*(cost(:,1)+beta*Vprev)), ...
	sum(prob2(1,:)'.*(cost(:,2)+beta*Vprev))]);

% Do it for state 2:
[~, pol2] = min( ...
	[sum(prob1(2,:)'.*(cost(:,1)+beta*Vprev)), ...
	sum(prob2(2,:)'.*(cost(:,2)+beta*Vprev))]);

pol_next = [pol1; pol2];
end
