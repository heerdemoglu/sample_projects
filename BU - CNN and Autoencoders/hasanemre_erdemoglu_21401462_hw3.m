% *************************************************************************
% EEE 443 - Neural Networks - Assignment 3
% Hasan Emre Erdemoglu - 21401462
% Due: Dec, 15 2019; 23.55 PM
% *************************************************************************
function hasanemre_erdemoglu_21401462_hw3(question)
clc
close all

switch question
	case '1'
		disp('Question 1:')
		question1();
	case '2'
		disp('Question 2:')
		% This question normally does not have a MATLAB counterpart.
		question2();
end

end

%% Question Functions:
function question1()
%% Initialize & Explain data:
load('assign3_data1.mat'); clear xForm; clear invXForm;
disp('Question 3 Outputs:');
disp(['Data is of size ', num2str(size(data,1)), ' by ', ...
	num2str(size(data,1)), '.']);
disp(['There are ', num2str(size(data,3)), ' channels.']);
disp(['There are ', num2str(size(data,4)), ' samples.']);
disp('---------------------------');

%% Part A:
% Preprocess to grayscale w/ model: (Y = 0.2126*R + 0.7152*G + 0.0722*B)
R = squeeze(data(:,:,1,:)); G = squeeze(data(:,:,2,:));
B = squeeze(data(:,:,3,:)); ndata = 0.2126*R + 0.7152*G + 0.0722*B;
clear R; clear G; clear B; % Workspace cleanup

% Calculate mean and std to clip & do normalization
mInts = mean(ndata,3); stdInts = std(ndata,[],3);
ndata = ndata - repmat(mInts,1,1,10240); % mean adjustment
clear mInts; % cleanup

% Do a clip mask and reflect it on dataset:
clip = repmat(3*stdInts, 1 ,1, 10240); 
negclip = ndata < -clip; posclip = ndata > clip;
ndata = ndata .* ~negclip + ~negclip .*-clip;
ndata = ndata .* ~posclip + ~posclip .*-clip;

% % Old code clips to 0 not to 3-std values
% ndata(ndata < -clip) = -clip;
% ndata(ndata < -clip) = 0; ndata(ndata > clip) = 0;

clear stdInts; clear clip;

% Now rescale:
ndata = rescale(ndata, 0.1, 0.9);

% Display random patches:
plotRandomIms(data, ndata);

%% Part B: - Weight init, uniform Xavier
% Find beta and rho that works well: - I checked some values manually,
% then set the tests around these values as grid search. global optima
% may not reside here more experiments on a wider range is needed
betas = linspace(0.0001,0.01,10); % 10
rhos = linspace(0.1,0.3,5); % 5
[params_b] = generateParamSet(64, 5e-4, betas, rhos , 1);

% Flatten input:
ndata = reshape(ndata, [], 10240); % Reshape to size 256 by 10240

disp('Doing a grid search over beta and rho values:');
% Convert to double (must for doing the calculations)
ndata = double(ndata); data = double(data);

% Test on params:
[weStar, bestBeta, bestRho, betastar, rhostar] = ...
	partB_experiments(params_b, ndata, betas, rhos);

%% Part C:
paramStar = struct('Lin', 256, 'Lhid', 64, 'lambda',5e-4, ...
	'beta', betas(betastar), 'rho', rhos(rhostar));

% Deconstruct best weights - already calculated:
[w1st, w2st, ~, ~] = unwrapper(weStar, paramStar.Lin, paramStar.Lhid);

% Display weights:
dispWeights(w1st, 8,8);

%% Part D:
% Retrain for different values: - 9 experiments in total
Lhid_exp = linspace(10,100,3);
lambda_exp = linspace(0,0.10e-3,3);
[params_d] = generateParamSet(Lhid_exp, lambda_exp, ...
	bestBeta, bestRho , []);

% Outputs cell array to deal with multi-dim array of different sizes.
[weightsAll] = partD_experiments(params_d, ndata);

% Draw images:
for i = 1:size(params_d,2)
	dispWeights(weightsAll{i},ceil(sqrt(params_d(i).Lhid)), ...
		ceil(sqrt(params_d(i).Lhid)));
	
	% Beautify plot by adding title: - overrides old title.
	sgtitle(['Hidden layer features with lambda = ', ...
		num2str(params_d(i).lambda), ...
		' and Lhid = ', num2str(params_d(i).Lhid)]);
	
end
end

function question2()
disp('This question is implemented in Phyton by using given environment.');
disp('No output is available.');
end

%% Question 1 - Helper Functions:
function [J, Jgrad] = aeCost(we,data,params)
% MATLAB doc suggests Jgrad should be given with nargout
N = size(data,2); a1 = data; % Let a1 = data for ease of reading

% Extract data from given parameters:
Lin = params.Lin; Lhid = params.Lhid; lambda = params.lambda;
beta = params.beta; rho = params.rho;

% Unwrap weights for building the autoencoder
[w1,w2,b1,b2] = unwrapper(we, Lin, Lhid);

% Do autoencoding to extract predictions:
[a2, dz2] = sigmoid(w1 * a1 + repmat(b1,1,N));
[a3, dz3] = sigmoid(w2 * a2 + repmat(b2,1,N));

% Calculate rho_b from avg hidden unit activations:
rho_b = mean(a2,2);

% Calculate KL term:
kl = (rho*log(rho./rho_b) + (1-rho)*log((1-rho)./(1-rho_b)));

% Now do the cost operation: - fminunc takes scalar cost (mean)
J = 1/2 * mean(sum(abs(data-a3).^2,1)) + ...
	lambda/2 * (sum(w1(:).^2) + sum(w2(:).^2)) + ...
	beta * sum(kl);

% Derivative calculation: KL term - rho has dependency on rho_b, hence a1
% -- w1,b1 terms Tykhonov term - dependency on w1, w2 terms Avg sq. error
% term - dependency on a2 -- w2,w1,b2,b1 terms Calculate backprop for
% squared term, then add respective partial derivs.

% KL derivative:
dKL = (-rho./rho_b + (1-rho)./(1-rho_b));
dKL = repmat(dKL, 1, N); % repeat for all samples

del3 = -(a1-a3) .* dz3;
dw2 = (del3 * a2') ./ N + lambda * w2;
db2 = sum(del3,2) ./ N;

del2 = (w2' * del3  + beta * dKL).* dz2;
dw1 = (del2 * a1') ./ N + lambda * w1;
db1 = sum(del2,2) ./ N;

if nargout > 1 % gradient required
	% Vectorize all weights:
	Jgrad = [dw1(:); dw2(:); db1(:); db2(:)];
end
end
function [w1,w2,b1,b2] = unwrapper(we, Lin, Lhid)
% Unwraps weight vector.
w_leng = Lin*Lhid;
w1 = we(1:w_leng); w1 = reshape(w1, Lhid, Lin);
w2 = we(w_leng+1:2*w_leng); w2 = reshape(w2, Lin, Lhid);
b1 = we(2*w_leng+1:2*w_leng+Lhid);
b2 = we(2*w_leng+Lhid+1:end);
end
function [o, do] = sigmoid(z)
% Calculates sigmoid activation for the neurons, z: activation signal
o = 1./(1+exp(-z));
do = o .* (1-o);
end
function [params] = generateParamSet(Lhids, lambdas, betas, rhos, choice)
if choice == 1
	% For Part B
	c = 1;
	for i = 1: length(betas)
		for j = 1: length(rhos)
			params(c) = struct('Lin', 256, 'Lhid', Lhids, 'lambda',5e-4, ...
				'beta', betas(i), 'rho', rhos(j));
			c = c+1;
		end
	end
else
	% For Part B:
	c = 1;
	for i = 1: length(Lhids)
		for j = 1: length(lambdas)
			params(c) = struct('Lin', 256, 'Lhid', Lhids(i), 'lambda', ...
				lambdas(j),'beta', betas, 'rho', rhos);
			c = c+1;
		end
	end
end
end
%% Flow functions: - Divide and Conquer, ease of readibility and workspace
function plotRandomIms(data, ndata)
disp('Plotting figures, imshow is slow, this will take some time.');
ranidx = randperm(10240, 200);
figure;
for i = 1:200
	subplot(10,20,i);
	imshow(data(:,:,:,ranidx(i)));
end

% Beautify plot by adding title: -- Works after 2018b, comment if fails.
sgtitle('Random sample patches in RGB format');

figure;
for i = 1:200
	subplot(10,20,i);
	imshow(ndata(:,:,ranidx(i)));
end

% Beautify plot by adding title: -- Works after 2018b, comment if fails.
sgtitle('Random sample patches in normalized version');
end
function [weStar, bestBeta, bestRho, betastar, rhost] = ...
	partB_experiments(params, ndata, betas, rhos)
J_min = inf; J_min_idx = -1;  % for min test
for exps = 1: length(params)
	
	% The weights and biases are symm. in 1 layer case so this is enough:
	wo = sqrt(6/(params(exps).Lhid+params(exps).Lin));
	
	w1 = -wo + rand(params(exps).Lhid,params(exps).Lin) * 2 * wo;
	w2 = w1'; % tied weights
	b1 = -wo + rand(params(exps).Lhid,1) * 2 * wo;
	b2 = -wo + rand(params(exps).Lin,1) * 2 * wo;
	
	we = [w1(:); w2(:); b1; b2];
	
	% Convert everything to double
	we = double(we);
	
	disp(['Doing experiments, please wait ... ', num2str(exps), '/', ...
		num2str(length(params)), '.']) ;
	
	% USE fminunc to do gradient descent: - minimum unconstrainted
	opts = optimset('GradObj','on','MaxIter',100); % used in fmincg
	fcn = @(w)aeCost(w,ndata,params(exps));
	[weFinal] = fmincg(fcn, we, opts); % Conj.Gradient Descent
	% I asked this to the course asistant, I was not able to use fminunc as
	% hessian matrix is of 33088 by 33088 size. Question asks for a solver
	% hence I used this one.
	
	% Now evaluate to see the cost and store it, always keep the minimum
	% cost:
	[J(exps),~] = feval(fcn,weFinal);
	disp(['Cost value = ' , num2str(J(exps)), '.']);
	if J_min > J(exps)
		J_min = J(exps);
		J_min_idx = exps;
		weStar = weFinal;
	end
	
end

% Find best beta, rho:
betastar = ceil(J_min_idx / length(rhos));
if mod(J_min_idx,length(rhos)) == 0
	rhost = length(rhos);
else
	rhost = mod(J_min_idx,length(rhos));
end

Jmap = reshape(J, length(betas), length(rhos));

% Display best results:
disp(['Best cost found in index: ', num2str(J_min_idx), ', w/ cost: ', ...
	num2str(J_min), '.']);
disp(['Best beta value is: ', num2str(betas(betastar)), ...
	', w/ index ' , num2str(betastar) , '.']);
disp(['Best rho value is: ', num2str(rhos(rhost)), ', w/ index ' , ...
	num2str(rhost) , '.']);

bestBeta = betas(betastar);
bestRho = rhos(rhost);

figure; imagesc(Jmap); title(' 2D Experiment Cost Values (Jmap)');
xlabel('rho indices'), ylabel('beta indices'); colorbar;
disp('Note that Jmap can be used to do further refinements on params.');

% *************************************************************************
% NOTE: The solver used in this question is appended below. It is taken
% from the link provided below: (broken into two lines, modified)
% https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/
% submissions/56393/versions/1/previews/mathwork/fmincg.m/index.html
% *************************************************************************
end
function dispWeights(w, xsize, ysize)
% Display first layer connection weights: -- encoder weights
w = reshape(w, [], 16, 16);

% Plot all images:
figure;
for i = 1:size(w,1)
	subplot(xsize,ysize,i);
	imagesc(squeeze(w(i,:,:))); colormap('gray');
end
% Beautify plot by adding title: -- Works after 2018b, comment if fails.
sgtitle('Encoding Weights for the Hidden Layer');
end
function [weightsAll] = partD_experiments(params, ndata)
disp('Doing a grid search over Hidden layer size and lambda values:');
% Test on params:
for exps = 1: length(params)
	
	% The weights and biases are symm. in 1 layer case so this is enough:
	wo = sqrt(6/(params(exps).Lhid+params(exps).Lin));
	
	w1 = -wo + rand(params(exps).Lhid,params(exps).Lin) * 2 * wo;
	w2 = w1'; % tied weights
	b1 = -wo + rand(params(exps).Lhid,1) * 2 * wo;
	b2 = -wo + rand(params(exps).Lin,1) * 2 * wo;
	
	we = [w1(:); w2(:); b1; b2];
	
	% Convert everything to double
	we = double(we);
	
	disp(['Doing experiments, please wait ... ', num2str(exps), '/', ...
		num2str(length(params)), '.']) ;
	
	% USE fminunc to do gradient descent: - minimum unconstrainted
	opts = optimset('GradObj','on','MaxIter',100); % used in fmincg
	fcn = @(w)aeCost(w,ndata,params(exps));
	[weFinal] = fmincg(fcn, we, opts); % Conj.Gradient Descent
	
	% Now evaluate to see the cost and store it, always keep the minimum
	% cost:
	[J(exps),~] = feval(fcn,weFinal);
	
	disp(['Cost value = ' , num2str(J(exps)), '.']);
	% I asked this to the course asistant, I was not able to use fminunc as
	% hessian matrix is of 33088 by 33088 size. Question asks for a solver
	% hence I used this one.
	
	% We just want to print out hidden layer features/weights so unwrap and
	% send only that information for display:
	% Deconstruct best weights - already calculated:
	[w1st, w2st, ~, ~] = unwrapper(weFinal, params(exps).Lin, params(exps).Lhid);
	
	
	weightsAll{exps} = w1st;
	
	% 	% Deconstruct best weights - already calculated:
	% 	[w1st, w2st, ~, ~] = unwrapper(weFinal, params(exps).Lin, ...
	% 		params(exps).Lhid);
	%
	% 	% Display first layer connection weights: -- encoder weights
	% 	w1_disp = reshape(w1st, params(exps).Lhid, 16, 16);
	%
	% 	% Plot all images:
	% 	figure;
	% 	for i = 1:size(w1_disp,1)
	% 		subplot(10,10,i);
	% 		imagesc(squeeze(w1_disp(i,:,:))); colormap('gray');
	% 	end
	%
	% 	% Beautify plot by adding title: - Works after 2018b, comment if fails.
	% 	sgtitle(['Encoding Weights for the Hidden Layer Test ', num2str(i)]);
	
end
end

%% External Code, references given:
function [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate function. Starting point
% is given by "X" (D by 1), and the function named in the string "f", must
% return a function value and a vector of partial derivatives. The Polack-
% Ribiere flavour of conjugate gradients is used to compute search directions,
% and a line search using quadratic and cubic polynomial approximations and the
% Wolfe-Powell stopping criteria is used together with the slope ratio method
% for guessing initial step sizes. Additionally a bunch of checks are made to
% make sure that exploration is taking place and that extrapolation will not
% be unboundedly large. The "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The function returns when either its length is up, or if no further
% progress can be made (ie, we are at a minimum, or so close that due to
% numerical problems, we cannot get any closer). If the function terminates

% within a few iterations, it """""could be""""" an indication that the function value
% and derivatives are not consistent****pas coherentes entres elles (ie, there may be a bug in the
% implementation of your "f" function). The function returns the found

% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
%
% See also: checkgrad
%
% Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
%
%
% (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
%
% Permission is granted for anyone to copy, use, or modify these
% programs and accompanying documents for purposes of research or
% education, provided this copyright notice is retained, and note is
% made of any changes that have been made.
%
% These programs and documents are distributed without any warranty,
% express or implied.  As the programs were written for research
% purposes only, they have not been tested to the degree that would be
% advisable in any important application.  All use of these programs is
% entirely at the user's own risk.
%
% [ml-class] Changes Made:
% 1) Function name and argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && ...
		isfield(options, 'MaxIter')
	length = options.MaxIter;
else
	length = 100;
end


RHO = 0.01;                       % a bunch of constants for line searches
SIG = 0.5;  % RHO and SIG are the constants in the Wolfe-Powell conditions
INT = 0.1; % don't reeval within 0.1 of the limit of the current bracket
EXT = 3.0;               % extrapolate maximum 3 times the current bracket
MAX = 20;                    % max 20 function evaluations per line search
RATIO = 100;                                 % maximum allowed slope ratio

argstr = ['feval(f, X'];        % compose string used to call function
for i = 1:(nargin - 3)
	argstr = [argstr, ',P', int2str(i)];
end
argstr = [argstr, ')'];

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
S=['Iteration '];

i = 0;                                      % zero the run length counter
ls_failed = 0;                        % no previous line search has failed
fX = [];
[f1 df1] = eval(argstr);                 % get function value and gradient
i = i + (length<0);                                   % count epochs?!
s = -df1;                                   % search direction is steepest
d1 = -s'*s;                                            % this is the slope
z1 = red/(1-d1);                     % initial step is red/(|s|+1)
%**douille


while i < abs(length)                                  % while not finished
	i = i + (length>0);                                  % count iterations?!
	% fprintf('test verif minim :  %f\n',i);
	X0 = X; f0 = f1; df0 = df1;               % make a copy of current values
	
	
	X = X + z1*s;                                        % begin line search
	[f2 df2] = eval(argstr);
	i = i + (length<0);                                    % count epochs?!
	d2 = df2'*s;
	f3 = f1; d3 = d1; z3 = -z1;        % initialize point 3 equal to point 1
	if length>0, M = MAX; else M = min(MAX, -length-i); end
	success = 0; limit = -1;                     % initialize quanteties
	while 1
		while ((f2 > f1+z1*RHO*d1) || (d2 > -SIG*d1)) && (M > 0)
			limit = z1;                                    % tighten the bracket
			if f2 > f1
				z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+f2-f3);            % quadratic fit
			else
				A = 6*(f2-f3)/z3+3*(d2+d3);                            % cubic fit
				B = 3*(f3-f2)-z3*(d3+2*d2);
				z2 = (sqrt(B*B-A*d2*z3*z3)-B)/A;   % numerical error possible - ok!
			end
			if isnan(z2) || isinf(z2)
				z2 = z3/2;           % if we had a numerical problem then bisect
			end
			z2 = max(min(z2, INT*z3),(1-INT)*z3); % don't acc too close to limits
			z1 = z1 + z2;                                     % update the step
			X = X + z2*s;
			[f2 df2] = eval(argstr);
			M = M - 1; i = i + (length<0);                    % count epochs?!
			d2 = df2'*s;
			z3 = z3-z2;              % z3 is now relative to the location of z2
		end
		if f2 > f1+z1*RHO*d1 || d2 > -SIG*d1
			break;                                         % this is a failure
		elseif d2 > SIG*d1
			success = 1; break;                                        % success
		elseif M == 0
			break;                                                    % failure
		end
		A = 6*(f2-f3)/z3+3*(d2+d3);                % make cubic extrapolation
		B = 3*(f3-f2)-z3*(d3+2*d2);
		z2 = -d2*z3*z3/(B+sqrt(B*B-A*d2*z3*z3));   % num. error possible - ok!
		if ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0
			% num prob or wrong sign?
			if limit < -0.5                         % if we have no upper limit
				z2 = z1 * (EXT-1);           % the extrapolate the maximum amount
			else
				z2 = (limit-z1)/2;                             % otherwise bisect
			end
		elseif (limit > -0.5) && (z2+z1 > limit) % extraplation beyond max?
			z2 = (limit-z1)/2;                                          % bisect
		elseif (limit < -0.5) && (z2+z1 > z1*EXT)  % extrapolation beyond limit
			z2 = z1*(EXT-1.0);                       % set to extrapolation limit
		elseif z2 < -z3*INT
			z2 = -z3*INT;
		elseif (limit > -0.5) && (z2 < (limit-z1)*(1.0-INT)) % too clos to lim?
			z2 = (limit-z1)*(1.0-INT);
		end
		f3 = f2; d3 = d2; z3 = -z2;             % set point 3 equal to point 2
		z1 = z1 + z2; X = X + z2*s;                 % update current estimates
		[f2 df2] = eval(argstr);
		M = M - 1; i = i + (length<0);                        % count epochs?!
		d2 = df2'*s;
	end                                                 % end of line search
	
	if success                                    % if line search succeeded
		f1 = f2; fX = [fX' f1]';
		% 		fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
		s = (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;  % Polack-Ribiere direction
		tmp = df1; df1 = df2; df2 = tmp;                    % swap derivatives
		d2 = df1'*s;
		if d2 > 0                                % new slope must be negative
			s = -df1;                      % otherwise use steepest direction
			d2 = -s'*s;
		end
		z1 = z1 * min(RATIO, d1/(d2-realmin));      % slope ratio but max RATIO
		d1 = d2;
		ls_failed = 0;                       % this line search did not fail
	else
		X = X0; f1 = f0; df1 = df0; % restore pnt from before failed search
		if ls_failed || i > abs(length)    % line search failed twice in a row
			break;                      % or we ran out of time, so we give up
		end
		tmp = df1; df1 = df2; df2 = tmp;                    % swap derivatives
		s = -df1;                                             % try steepest
		d1 = -s'*s;
		z1 = 1/(1-d1);
		ls_failed = 1;                              % this line search failed
	end
	if exist('OCTAVE_VERSION')
		fflush(stdout);
	end
end
%fprintf('\n');
%fprintf('%s %4i | Cost: %4.6e\r', S, i, f1);
end