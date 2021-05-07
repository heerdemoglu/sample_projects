% *************************************************************************
% EEE 443 - Neural Networks - Assignment 2
% Hasan Emre Erdemoglu - 21401462
% *************************************************************************
function hasanemre_erdemoglu_21401462_hw2(question)
clc
close all

switch question
	case '1'
		disp('Question 1:')
		% This question normally does not have a MATLAB counterpart.
		question1()
	case '2'
		disp('Question 2:')
		disp('Note that this question will use a lot of time.')
		disp('Note that the results will change from run to run,')
		disp('due to randomness of the dataset')
		disp('Please wait.')
		question2();
	case '3'
		disp('Question 3:')
		question3();
end

end

%% Question Functions:
function question1()
disp('No output is available.')
end
function question2()
%% Initialization:
% Load the dataset:
load assign2_data1.mat

% Dataset Specifications: *************************************************
% Labels: 2 classes, Cats(0) and Cars(1) - Binary classification problem
% Train set has 1900 images, Test set has 1000 images.
% Images are in unsigned 8-bit integer form. Range: (0,255)
% Image size is 32x32.
% *************************************************************************
[trainims, trainlbls, testims, testlbls] = ...
	preprocessDatasets(trainims, trainlbls,testims, testlbls);

%% Part A - Initialization: 
% Parameters to adjust: ***************************************************
% SINGLE LAYER PERCEPTRON CASE - NO MOMENTUM IMPLEMENTED
% Select learning rate as 0.1 to have more epochs -> More refined cost fcn.
% Adjust parameters: layer_size, batch_size
lr = 0.05; mu = 0; sig = 1;  max_epoch = 50;
batch_size = [25, 75, 225];
layer_size = [100, 650, 1500];
% *************************************************************************

% Show initialization histograms samples:
disp('All histogram initializations are samples for w1. It is same for');
disp('other parameters too.');
[wG, ~, ~, ~] = init_network_weights_1hl(mu, sig, 1024, 600, 1, 10, 1);
figure; histogram(wG); clear wG;
title('Histogram of Gaussian Weight Initialization with 0 mean, 1 std');

[wZ, ~, ~, ~] = init_network_weights_1hl(mu, sig, 1024, 600, 1, 10, 2);
figure; histogram(wZ); clear wZ;
title('Histogram of Zero Weight Initialization');

[wX, ~, ~, ~] = init_network_weights_1hl(mu, sig, 1024, 600, 1, 10, 3);
figure; histogram(wX); clear wX;
title('Histogram of Xavier Weight Initialization');

%% Part A - Implementation:
% Grid search on 3 batch and layer sizes and for 3 weight inits.
[best_initC_idx, best_cost, best_error] = ...
	experiment_on_hyperparameters(layer_size, batch_size, ...
	trainims, trainlbls, testims, testlbls, ...
	max_epoch, lr, mu, sig);

disp('Best layer size chosen:'); disp(layer_size(best_cost(1)));
disp('Best batch size chosen:'); disp(batch_size(best_cost(2)));

%% Retrain with the chosen set of hyperparameters: (finalize)
[w1, b1, w2, b2] = ...
	init_network_weights_1hl(mu, sig, 1024, ...
	layer_size(best_cost(1)), 1, ...
	batch_size(best_cost(2)), best_initC_idx);

[traCost, traError, testCost, testError, ...
	testPred, testFlat, ...
	~, ~, ~, ~] = ...
	single_layer_model(trainims, trainlbls, testims, ...
	testlbls, w1, w2, b1, b2, max_epoch, ...
	batch_size(best_cost(2)), lr);

%% Show results:
figure; hold on; title('MSE for Training & Test Data');
plot(traCost); plot(testCost); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

figure; hold on; title('MCE for Training & Test Data');
plot(traError); plot(testError); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

disp('Final test accuracy: ');
disp(sum(testPred==testFlat)./length(testPred));

%% Part C:
Nlow = ceil(layer_size(best_cost(1))*0.25);
Nhigh = ceil(layer_size(best_cost(1))*4);

disp('Nlow: '); disp(Nlow); disp('Nhigh: '); disp(Nhigh);

% Retrain with the chosen set of hyperparameters: (Nlow)
[w1, b1, w2, b2] = ...
	init_network_weights_1hl(mu, sig, 1024, ...
	Nlow, 1, ...
	batch_size(best_cost(2)), best_initC_idx);

[traCostNlow, traErrorNlow, testCostNlow, testErrorNlow, ...
	testPredNlow, testFlatNlow, ...
	~, ~, ~, ~] = ...
	single_layer_model(trainims, trainlbls, testims, ...
	testlbls, w1, w2, b1, b2, max_epoch, ...
	batch_size(best_cost(2)), lr);

% Show results:
figure; hold on; title('MSE for Training & Test Data (Nlow)');
plot(traCostNlow); plot(testCostNlow); grid on; %axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

figure; hold on; title('MCE for Training & Test Data (Nlow)');
plot(traErrorNlow); plot(testErrorNlow); grid on; %axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

disp('Final test accuracy: (Nlow)');
disp(sum(testPredNlow==testFlatNlow)./length(testPred));

% Retrain with the chosen set of hyperparameters: (Nhigh)
[w1, b1, w2, b2] = ...
	init_network_weights_1hl(mu, sig, 1024, ...
	Nhigh, 1, ...
	batch_size(best_cost(2)), best_initC_idx);

[traCostNhigh, traErrorNhigh, testCostNhigh, testErrorNhigh, ...
	testPredNhigh, testFlatNhigh, ...
	w1_fin, b1_fin, w2_fin, b2_fin] = ...
	single_layer_model(trainims, trainlbls, testims, ...
	testlbls, w1, w2, b1, b2, max_epoch, ...
	batch_size(best_cost(2)), lr);

% Show results:
figure; hold on; title('MSE for Training & Test Data (Nhigh)');
plot(traCostNhigh); plot(testCostNhigh); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

figure; hold on; title('MCE for Training & Test Data (Nhigh)');
plot(traErrorNhigh); plot(testErrorNhigh); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

disp('Final test accuracy: (Nhigh)');
disp(sum(testPredNhigh==testFlatNhigh)./length(testPred));

% Part C - Comparison:
figure; 
plot(traCostNlow); hold on; plot(traCost); plot(traCostNhigh);
xlabel('Epochs'); ylabel('MSE Loss'); grid on;
legend('Train - Nlow', 'Train - Nstar', 'Train - Nhigh');
title('Nlow, Nstar & Nhigh MSE for train sets');

figure; hold on;
plot(testCostNlow); plot(testCost); plot(testCostNhigh); grid on;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Test - Nlow', 'Test - Nstar', 'Test - Nhigh');
 title('Nlow, Nstar & Nhigh MSE for test sets');

figure; 
plot(traErrorNlow); hold on; plot(traError); plot(traErrorNhigh);
xlabel('Epochs'); ylabel('MCE Loss'); grid on;
legend('Train - Nlow', 'Train - Nstar', 'Train - Nhigh');
title('Nlow, Nstar & Nhigh MCE for train sets');

figure; hold on;
plot(testErrorNlow); plot(testError); plot(testErrorNhigh); grid on;
xlabel('Epochs'); ylabel('MCE Loss'); 
legend('Test - Nlow', 'Test - Nstar', 'Test - Nhigh');
 title('Nlow, Nstar & Nhigh MCE for test sets');

%% Part D: - Two layer Model - No momentum implementation
% This time I won't be doing experiments. I did it myself by picking
% values. (To save time when running this script.)
% Xavier init will be used.
hl1size = 300; hl2size = 150; % lr, max_epoch, mu, sig carried over
inSize = 1024; outSize = 1;
batchSize = batch_size(2);
[w1,b1,w2,b2,w3,b3] = init_network_weights_2hl(mu, sig, ...
	inSize, hl1size, hl2size, outSize, ...
	batchSize, 3); % xavier init

% Copy same parameters for part e:
w1e = w1; w2e= w2; w3e = w3; b1e = b1; b2e = b2; b3e = b3;

[traCost2hl, traError2hl, testCost2hl, testError2hl, ...
	testPred2hl, testFlat2hl] = ...
	two_layer_model(trainims, trainlbls, testims, testlbls, ...
	w1, w2, w3, b1, b2, b3, ...
	max_epoch, batchSize, lr);

% Show results:
figure; hold on; title('MSE for Training & Test Data (2HL w/o momentum)');
plot(traCost2hl); plot(testCost2hl); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

figure; hold on; title('MCE for Training & Test Data (2HL w/o momentum)');
plot(traError2hl); plot(testError2hl); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

disp('Final test accuracy: (2 HL no momentum)');
disp(sum(testPred2hl==testFlat2hl)./length(testPred2hl));

%% Part E: - Two layer Model - With momentum implementation
mr = 0.2; % assume momentum rate
[traCost2hle, traError2hle, testCost2hle, testError2hle, ...
	testPred2hle, testFlat2hle] = ...
	two_layer_model_momentum(trainims, trainlbls, testims, testlbls, ...
	w1e, w2e, w3e, b1e, b2e, b3e, ...
	max_epoch, batchSize, lr, mr);

% Show results:
figure; hold on; title('MSE for Training & Test Data (2HL w/ momentum)');
plot(traCost2hle); plot(testCost2hle); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

figure; hold on; title('MCE for Training & Test Data (2HL w/ momentum)');
plot(traError2hle); plot(testError2hle); grid on; axis tight;
xlabel('Epochs'); ylabel('MSE Loss'); 
legend('Training', 'Test');

disp('Final test accuracy: (2 HL w/ momentum)');
disp(sum(testPred2hle==testFlat2hle)./length(testPred2hle));

end
function question3()
%% Question 3:
clear; close all; clc; % For debug

%% Initialization:
% Load the dataset:
load assign2_data2.mat
disp('Initializing...');
% Dataset Specifications: *************************************************
% Task: Predict 4th word, based on previous 3 terms.
% Vocabulary/Class Size = 250 words
% Words: 1x250 cell array containing words in indexed order
% trainx: 3x372500 matrix, 372500 samples of trigrams
% traind: 1x372500 vector, 4th word of given trigram
%
% (testx,testd): Test set, same 3 word input, 1 output word format
% It has 46500 samples available.
%
% (valx,vald): Validation set, same 3 word input, 1 output word format
% It has 46500 samples available.
% *************************************************************************

%% Data Preprocessing:
% Shuffle the datasets:
shuff_tr_idx = randperm(372500, 372500);
trainx = trainx(:,shuff_tr_idx); traind = traind(shuff_tr_idx);
clear shuff_tr_idx; % clean up

shuff_va_idx = randperm(46500, 46500);
valx = valx(:, shuff_va_idx); vald = vald(shuff_va_idx);
clear shuff_va_idx; % clean up

shuff_te_idx = randperm(46500,46500);
testx = testx(:, shuff_te_idx); testd = testd(shuff_te_idx);
clear shuff_te_idx; % clean up

%% One hot encode datasets:
trainx = onehot_encode(trainx, 250); traind = onehot_encode(traind, 250);
valx = onehot_encode(valx, 250); vald = onehot_encode(vald, 250);
testx = onehot_encode(testx, 250); testd = onehot_encode(testd, 250);

% Squeeze unnecessary dimension on the labels:
traind = squeeze(traind); vald = squeeze(vald); testd = squeeze(testd);

% %% Merge onehot words into a single vector:
% trainx = reshape(trainx, [], 372500); valx = reshape(valx, [], 46500);
% testx = reshape(testx, [], 46500);


%% Divide to batches: - Ignore non-full batches
trainx = reshape(trainx(:,:,1:372400), 3, [], 200, 1862);
traind = reshape(traind(:,1:372400), [], 200, 1862);
valx = reshape(valx(:,:,1:46400), 3, [], 200, 232);
vald = reshape(vald(:,1:46400), [], 200, 232);
% testx = reshape(testx(:,:,1:46400), 3, [], 200, 232);
% testd = reshape(testd(:,1:46400), [], 200, 232);


%% Given parameters in the Question:
% Early stop: Based on X-Entropy on validation data.
lr = 0.15; mr = 0.85; max_epoch = 50;
mu = 0; sigma = 0.01;

% Weight matrices as given in the Question: -Experiment on D and P values.
D = [32, 16, 8]; P = [256, 128, 64];

%% Part A:
% Initialize weights:
wew1 = repmat(normrnd(mu, sigma, [250,3*D(1)]), 3, 1); % 750x96
ethw1 = normrnd(mu, sigma, [3*D(1),P(1)]);
ethb1 = normrnd(mu, sigma, [P(1), 1]);
htow1 = normrnd(mu, sigma, [P(1), 250]);
htob1 = normrnd(mu, sigma, [250, 1]);

wew2 = repmat(normrnd(mu, sigma, [250,3*D(2)]), 3, 1); % 750x96
ethw2 = normrnd(mu, sigma, [3*D(2),P(2)]);
ethb2 = normrnd(mu, sigma, [P(2), 1]);
htow2 = normrnd(mu, sigma, [P(2), 250]);
htob2 = normrnd(mu, sigma, [250, 1]);

wew3 = repmat(normrnd(mu, sigma, [250,3*D(3)]), 3, 1); % 750x96
ethw3 = normrnd(mu, sigma, [3*D(3),P(3)]);
ethb3 = normrnd(mu, sigma, [P(3), 1]);
htow3 = normrnd(mu, sigma, [P(3), 250]);
htob3 = normrnd(mu, sigma, [250, 1]);

%% Train the networks:
disp('Training network 1: ...')
[htowStar1, htobStar1, ethwStar1, ethbStar1, wewStar1, epTr1, epVal1] = ...
	model(trainx, traind, valx, vald, htow1, htob1, ethw1, ethb1, wew1, ...
	max_epoch, mr, lr);

disp('Training network 2: ...')
[htowStar2, htobStar2, ethwStar2, ethbStar2, wewStar2, epTr2, epVal2] = ...
	model(trainx, traind, valx, vald, htow2, htob2, ethw2, ethb2, wew2, ...
	max_epoch, mr, lr);

disp('Training network 3: ...')
[htowStar3, htobStar3, ethwStar3, ethbStar3, wewStar3, epTr3, epVal3] = ...
model(trainx, traind, valx, vald, htow3, htob3, ethw3, ethb3, wew3, ...
	max_epoch, mr, lr);

%% Plot Costs:
figure; plot(epTr1); hold on; plot(epVal1); grid on; axis tight;
xlabel('Epochs'); ylabel('Cross Entropy Loss'); title('Network 1 Loss');
legend('Training','Validation');

figure; plot(epTr2); hold on; plot(epVal2); grid on; axis tight;
xlabel('Epochs'); ylabel('Cross Entropy Loss'); title('Network 2 Loss');
legend('Training','Validation');

figure; plot(epTr3); hold on; plot(epVal3); grid on; axis tight;
xlabel('Epochs'); ylabel('Cross Entropy Loss'); title('Network 3 Loss');
legend('Training','Validation');

%% Part A - Accuracy:
testx = double(testx); testd = double(testd);
[testPred] = predict(testx, wew1, ethw1, ethb1, htow1, htob1, 46500);

[~, idxtest] = max(testPred,[],1);
tpred = zeros(size(testPred));
tpred(idxtest,:) = 1;

accTest = sum(sum(tpred~=testd,1)/250)/46500;
disp(['Testing accuracy Network 1: ', num2str(accTest)]);

% [testPred] = predict(testx, wew2, ethw2, ethb2, htow2, htob2, 46500);
% 
% [~, idxtest] = max(testPred);
% tpred = zeros(size(testPred)); tpred(:,idxtest) = 1;
% 
% accTest = sum(tpred==testd);
% disp(['Testing accuracy Network 2: ', num2str(accTest)]);
% 
% [testPred] = predict(testx, wew3, ethw3, ethb3, htow3, htob3, 46500);
% 
% [~, idxtest] = max(testPred);
% tpred = zeros(size(testPred)); tpred(:,idxtest) = 1;
% 
% accTest = sum(tpred==testd);
% disp(['Testing accuracy Network 3: ', num2str(accTest)]);

%% Part B:
% index = randperm(46500,5); % pick 5 samples
% sampx = testx(:,index); sampd = testd(index); clear index;
% sampx = onehot_encode(sampx, 250); sampd = onehot_encode(sampd, 250);
% 
% % % Do predictions using models
% [sampPred1] = predict(sampx, wew1, ethw1, ethb1, htow1, htob1,5);
% [sampPred1, idxPred1] = sort(sampPred1, 'descend'); 

end

%% Helper Functions:
% Question 2 Helpers:
function [tri, trl, tei, tel] = preprocessDatasets(trainims, trainlbls, ...
	testims, testlbls)
% Data pre-processing:
% Normalize to continious range 0-1
% Normalize labels to range -1 and +1 (since we use tanh, using full range)
trainims = double(trainims) ./ 255; testims = double(testims) ./ 255;
trainlbls(trainlbls == 0) = -1; testlbls(testlbls == 0) = -1;

% Flatten images:
trainims = reshape(trainims, 1024, 1900);
testims = reshape(testims, 1024, 1000);

% Shuffle data:
shuffleidxs = ceil(randperm(1900,1900));
trims = trainims(:,shuffleidxs); trainims = trims;
trlbls = trainlbls(shuffleidxs); trainlbls = trlbls;
clear trims; clear trlbls; clear shuffleidxs;

shuffleidxs = ceil(randperm(1000,1000));
tims = testims(:,shuffleidxs); testims = tims;
tlbls = testlbls(shuffleidxs); testlbls = tlbls;
clear tims; clear tlbls; clear shuffleidxs;
% *************************************************************************
% Fix indices of the dataset for processing:
tri = trainims; trl = trainlbls;
tei = testims; tel = testlbls;

end
function [best_initC_idx, best_cost, best_error] = ...
	experiment_on_hyperparameters(layer_size, batch_size, ...
	trainims, trainlbls, testims, testlbls, ...
	max_epoch, lr, mu, sig)

for bt = 1:3 % batch test, hardcoded
	for lt = 1:3 % layer test, hardcoded
		disp('Doing tests on diffent batch & layer sizes, please wait...');
		% Generate parameters for testing:
		[w1G, b1G, w2G, b2G] = ...
			init_network_weights_1hl(mu, sig, 1024, layer_size(lt), 1, ...
			batch_size(bt), 1);
		[w1Z, b1Z, w2Z, b2Z] = ...
			init_network_weights_1hl(mu, sig, 1024, layer_size(lt), 1, ...
			batch_size(bt), 2);
		[w1X, b1X, w2X, b2X] = ...
			init_network_weights_1hl(mu, sig, 1024, layer_size(lt), 1, ...
			batch_size(bt), 3);
		
		% Do the training: -Only log final values achieved.
		[traCostGauss, traErrorGauss, testCostGauss, testErrorGauss, ...
			~, ~, ......
			~, ~, ~, ~] = ...
			single_layer_model(trainims, trainlbls, testims, testlbls, ...
			w1G, w2G, b1G, b2G, max_epoch, ...
			batch_size(bt), lr);
		
		% Keep necessary logs - We will return back to the best model
		traCostGaussLog(lt,bt) = traCostGauss(end);  
		traErrorGaussLog(lt,bt) = traErrorGauss(end);
		testCostGaussLog(lt,bt) = testCostGauss(end); 
		testErrorGaussLog(lt,bt) = testErrorGauss(end); 
		
		[traCostZero, traErrorZero, ...
			valCostZero, valErrorZero, ...
			~, ~, ~, ~, ~, ~] = ...
			single_layer_model(trainims, trainlbls, testims, ...
			testlbls, w1Z, w2Z, b1Z, b2Z, max_epoch, ...
			batch_size(bt), lr);
		
		% Keep necessary logs - We will return back to the best model
		traCostZeroLog(lt,bt) = traCostZero(end); 
		traErrorZeroLog(lt,bt) = traErrorZero(end);
		valCostZeroLog(lt,bt) = valCostZero(end);
		valErrorZeroLog(lt,bt) = valErrorZero(end); 

		[traCostXavi, traErrorXavi, ...
			valCostXavi, valErrorXavi, ...
			~, ~, ~, ~, ~, ~] = ...
			single_layer_model(trainims, trainlbls, testims, ...
			testlbls, w1X, w2X, b1X, b2X, max_epoch, ...
			batch_size(bt), lr);
		
		% Keep necessary logs - We will return back to the best model
		traCostXaviLog(lt,bt) = traCostXavi(end); 
		traErrorXaviLog(lt,bt) = traErrorXavi(end); 
		valCostXaviLog(lt,bt) = valCostXavi(end); 
		valErrorXaviLog(lt,bt) = valErrorXavi(end); 
	end
end

%% Pick the best values: - From validation data

% Check Gaussian Logs:
[min_valGauss_cost] = find_min_matrix(testCostGaussLog);
[min_valGauss_error] = find_min_matrix(testErrorGaussLog);
final_Gauss_cost = testCostGaussLog(min_valGauss_cost(1), ...
	min_valGauss_cost(2));
final_Gauss_error = testCostGaussLog(min_valGauss_error(1), ...
	min_valGauss_error(2));

% Check Zero Logs:
[min_valZero_cost] = find_min_matrix(valCostZeroLog);
[min_valZero_error] = find_min_matrix(valErrorZeroLog);
final_Zero_cost = valCostZeroLog(min_valZero_cost(1), ...
	min_valZero_cost(2));
final_Zero_error = valCostZeroLog(min_valZero_error(1), ...
	min_valZero_error(2));

% Check Xavier Logs:
[min_valXavi_cost] = find_min_matrix(valCostXaviLog);
[min_valXavi_error] = find_min_matrix(valErrorXaviLog);
final_Xavi_cost = valCostXaviLog(min_valXavi_cost(1), ...
	min_valXavi_cost(2));
final_Xavi_error = valCostXaviLog(min_valXavi_error(1), ...
	min_valXavi_error(2));

% Merge cost and error metrics to export to main function:
min_cost = [min_valGauss_cost;min_valZero_cost;min_valXavi_cost];
min_error = [min_valGauss_error;min_valZero_error;min_valXavi_error];

% Pick the best init type:
final_cost = [final_Gauss_cost,final_Zero_cost,final_Xavi_cost];
[~,best_initC_idx] = min(final_cost);
final_error = [final_Gauss_error,final_Zero_error,final_Xavi_error];
[~,best_initE_idx] = min(final_error);

% Best cost and error:
best_cost = min_cost(best_initC_idx,:);
best_error = min_error(best_initE_idx,:);

disp('Best init in terms of cost is: ');
disp(best_initC_idx);

disp('Note that 1 is Gaussian, 2 is Zero and 3 is Xavier initialization.');
disp('For Part B: These values are expected to be the same as mean');
disp('squared error also checks difference between output and labels');
disp('MCE compares equality of predictions after output is classfied');
disp('to a label, MSE does it before classification, namely, with');
disp('activations at the output layer.');

%% Display log results:
figure; imagesc(testCostGaussLog); colorbar;
title('Test on Gaussian Validation MSE'); xlabel('Layer Size Indices'); 
ylabel('Batch Size Indices');

figure; imagesc(testErrorGaussLog); colorbar;
title('Test on Gaussian Validation MCE'); xlabel('Layer Size Indices'); 
ylabel('Batch Size Indices');

figure; imagesc(valCostZeroLog); colorbar;
title('Test on Zero Validation MSE'); xlabel('Layer Size Indices'); 
ylabel('Batch Size Indices');

figure; imagesc(valErrorZeroLog); colorbar;
title('Test on Zero Validation MCE'); xlabel('Layer Size Indices'); 
ylabel('Batch Size Indices');

figure; imagesc(valCostXaviLog); colorbar;
title('Test on Xavier Validation MSE'); xlabel('Layer Size Indices'); 
ylabel('Batch Size Indices');

figure; imagesc(valErrorXaviLog); colorbar;
title('Test on Xavier Validation MCE'); xlabel('Layer Size Indices'); 
ylabel('Batch Size Indices');
end

function [A_curr, Z_curr] = forward_layer(A_prev, W, b)
% Calculates the activation signal and tanh output for given layer.
Z_curr = W * A_prev + b;
A_curr = tanh(Z_curr);
end
function [dCost_dw_1, dCost_db_1, del_0] = ...
	backprop_layer(z_1, a_1, del_1, w_1)
% Calculates backpropagation for the given layer.

% del_1 is the backpropagation error from next layer
temp = del_1 .* gradient_tanh(z_1);

% please see report for derivation
dCost_dw_1 = temp * a_1' ./ size(a_1,2);
dCost_db_1 = 1 * (del_1);
del_0 = w_1' * temp / size(temp,1);

end
function [dZ] = gradient_tanh(actv)
% Calculates the gradient of the tanh activation at dA's layer.
dZ = 1-(tanh(actv)).^2;
end

function [minidx] = find_min_matrix(A)
minimum=min(min(A));
[x,y]=find(A==minimum);

minidx = [x, y];
end

function [w1,b1,w2,b2] = ...
	init_network_weights_1hl(mu, sigma, inSize, hlSize, outSize, ...
	batchSize, init_type)
% Takes mu, sigma for Gaussian initialization.
% w: First dim is for current input's dimensionality, second dim is for the
% sizing of the layer which the weights are fed to.
% b: First dim is fixed to 1. Second dim is for the sizing of the layer
% which the bias is fed to.

% This function can be written in tensor form to accomodate automatic
% initialization for arbitrary size of layers, however since the question
% only delves until 2 HL MLP, I will skip that for now.

if (init_type == 1) % Gaussian initialization (1st next size, then prev)
	w1 = normrnd(mu, sigma, [hlSize, inSize]);
	b1 = zeros(hlSize, 1);
	w2 = normrnd(mu, sigma, [outSize, hlSize]);
	b2 = zeros(1, 1);
elseif (init_type  == 2) % Zero initialization
	w1 = zeros(hlSize, inSize);
	b1 = zeros(hlSize, 1);
	w2 = zeros(outSize, hlSize);
	b2 = zeros(1, 1);
else % Xavier initialization
	w1 = -sqrt(6/(1024+hlSize))/2 + ...
		rand(hlSize, inSize) .* sqrt(6/(1024+hlSize));
	w2 = -sqrt(6/(1024+hlSize))/2 + ... 
		rand(outSize, hlSize) .* sqrt(6/(1024+1));	
	b1 = zeros(hlSize, 1);
	b2 = zeros(1, 1);
end
end
function [w1,b1,w2,b2,w3,b3] = ...
	init_network_weights_2hl(mu, sigma, ...
	inSize, hl1Size, hl2Size, outSize, ...
	batchSize, init_type)
% Takes mu, sigma for Gaussian initialization.
% w: First dim is for current input's dimensionality, second dim is for the
% sizing of the layer which the weights are fed to.
% b: First dim is fixed to 1. Second dim is for the sizing of the layer
% which the bias is fed to.

% This function can be written in tensor form to accomodate automatic
% initialization for arbitrary size of layers, however since the question
% only delves until 2 HL MLP, I will skip that for now.

if (init_type == 1) % Gaussian initialization (1st next size, then prev)
	w1 = normrnd(mu, sigma, [hl1Size, inSize]);
	b1 = zeros(hl1Size, 1);
	w2 = normrnd(mu, sigma, [hl2Size, hl1Size]);
	b2 = zeros(hl2Size, 1);
	w3 = normrnd(mu, sigma, [outSize, hl2Size]);
	b3 = zeros(1, 1);
elseif (init_type  == 2) % Zero initialization
	w1 = zeros(hl1Size, inSize);
	b1 = zeros(hl1Size, 1);
	w2 = zeros(hl2Size, hl1Size);
	b2 = zeros(hl2Size, 1);
	w3 = zeros(outSize, hl2Size);
	b3 = zeros(1, 1);
else % Xavier initialization
	w1 = -sqrt(6/(inSize+hl1Size))/2 + ...
		rand(hl1Size, inSize) .* sqrt(6/(inSize+hl1Size));
	w2 = -sqrt(6/(hl1Size+hl2Size))/2 + ... 
		rand(hl2Size, hl1Size) .* sqrt(6/(hl1Size+hl2Size));
	w3 = -sqrt(6/(hl2Size+outSize))/2 + ... 
		rand(outSize, hl2Size) .* sqrt(6/(hl2Size+outSize));
	b1 = zeros(hl1Size, 1);
	b2 = zeros(hl2Size, 1);
	b3 = zeros(1, 1);
end
end

function [traCost, traError, testCost, testError, ...
	testPred, testFlat, ...
	w1_fin, b1_fin, w2_fin, b2_fin] = ...
	single_layer_model(trainims, trainlbls, testims, testlbls, ...
	w1, w2, b1, b2, max_epoch, batch_size, lr)

%% Pre generate batches - For train and validation sets: ******************
% Ignores non-full batch (Some data lost but not much, easier to implement)
traSetSize = length(trainlbls);
valSetSize = length(testlbls);

traExcess = mod(traSetSize,batch_size);
valExcess = mod(valSetSize,batch_size);

trainims = trainims(:,1:end-traExcess);
trainlbls = trainlbls(1:end-traExcess); % clear traExcess;

% Reshape layer:
trainims = reshape(trainims, 1024, batch_size, ...
	length(trainlbls)./batch_size);
trainlbls = reshape(trainlbls, batch_size, ...
	length(trainlbls)./batch_size);

testims = testims(:,1:end-valExcess);
testlbls = testlbls(1:end-valExcess); % clear valExcess;

% Reshape layer:
testims = reshape(testims, 1024, batch_size, ...
	length(testlbls)/batch_size);
testlbls = reshape(testlbls, batch_size, ...
	length(testlbls)/batch_size);
% *************************************************************************

for epoch = 1:max_epoch
	% Goes over entire dataset -> go over all batches
	for batch = 1: size(trainims,3)
		% Set 1 batch: - go through all images from start to end
		% Every item in the batch gets added to weight update automatically
		% thanks to the matrix operations
		im = trainims(:,:,batch); % This will be 1024x{batchSize}
		lbls = trainlbls(:,batch); % this will be {batchSize}x1
		
		% Forward Pass:
		[A1, Z1] = forward_layer(im, w1, b1); % {layerSize}x{batchSize}
		[A2, Z2] = forward_layer(A1, w2, b2); % 1x{batchSize}
		
		% Compute Mean Squared Cost: - Sums over batch samples by itself
		traCost(epoch) = 0.5 * mean((A2-lbls').^2);
		
		% Compute Mean Classification Error: (Between 0 and 1)
		traPred(A2 < 0) = -1; traPred(A2 >= 0) = 1;
		traError(epoch) = sum(lbls' ~= traPred) ./ batch_size;
		
		% Backward Pass:
		% Local gradient for output layer:
		del2 = (A2-lbls');
		[dCost_dw2, dCost_db2, del1] = backprop_layer(Z2, A1, del2, w2);
		[dCost_dw1, dCost_db1, ~] = backprop_layer(Z1, im, del1, w1);
		
		% Delta Rule Update:
		w2 = w2 - lr * dCost_dw2;
		b2 = b2 - lr * dCost_db2;
		
		w1 = w1 - lr * dCost_dw1;
		b1 = b1 - lr * dCost_db1;
	end
	
	%% After an epoch is done, calculate error on test set:
	% We need cross validation, using test like this is not ideal in my
	% opinion. However as the question asked it this way, I will do this
	% this way. Note that test set will be given the same batch size;
	% this is made for the equations to hold:
	for b = 1:size(testims,3)
		test_im = testims(:,:,b);
		test_lbls = testlbls(:,b);
		
		% Forward Pass:
		[val_A1, val_Z1] = forward_layer(test_im, w1, b1);
		[val_A2, val_Z2] = forward_layer(val_A1, w2, b2);
		
		val_preds(:,b) = val_A2;
		
		% Check costs:
		testCost(epoch) = mean((val_preds(:,b)-test_lbls).^2);
		
		% Pick classes:
		val_preds(val_preds >= 0) = 1;
		val_preds(val_preds < 0) = -1;
		
		testError(epoch) = sum(val_preds(:,b) ~= test_lbls) ./ ...
			batch_size;
	end
	
	% Flatten predictions: (as computing is now complete)
	testPred = reshape(val_preds, 1, size(val_preds,1) * ...
		size(val_preds,2));
	testFlat = reshape(testlbls, size(testPred,1), ...
		size(testPred,2));
	
% 	% Check test error should not increase
% 	if epoch >= 2 && (abs(testCost(epoch) - testCost(epoch-1)) ...
% 			./ abs(testCost(epoch-1))) < 0.001
% 		break;
% 	end
	
end
w1_fin = w1; b1_fin = b1; w2_fin = w2; b2_fin = b2;
end
function [traCost, traError, testCost, testError, ...
	testPred, testFlat] = ...
	two_layer_model(trainims, trainlbls, testims, testlbls, ...
	w1, w2, w3, b1, b2, b3, ...
	max_epoch, batch_size, lr)

%% Pre generate batches - For train and validation sets: ******************
% Ignores non-full batch (Some data lost but not much, easier to implement)
traSetSize = length(trainlbls);
valSetSize = length(testlbls);

traExcess = mod(traSetSize,batch_size);
valExcess = mod(valSetSize,batch_size);

trainims = trainims(:,1:end-traExcess);
trainlbls = trainlbls(1:end-traExcess);

% Reshape layer:
trainims = reshape(trainims, 1024, batch_size, ...
	length(trainlbls)./batch_size);
trainlbls = reshape(trainlbls, batch_size, ...
	length(trainlbls)./batch_size);

testims = testims(:,1:end-valExcess);
testlbls = testlbls(1:end-valExcess);

% Reshape layer:
testims = reshape(testims, 1024, batch_size, ...
	length(testlbls)/batch_size);
testlbls = reshape(testlbls, batch_size, ...
	length(testlbls)/batch_size);
% *************************************************************************

for epoch = 1:max_epoch
	% Goes over entire dataset -> go over all batches
	for batch = 1: size(trainims,3)
		% Set 1 batch: - go through all images from start to end
		% Every item in the batch gets added to weight update automatically
		% thanks to the matrix operations
		im = trainims(:,:,batch); % This will be 1024x{batchSize}
		lbls = trainlbls(:,batch); % this will be {batchSize}x1
		
		% Forward Pass:
		[A1, Z1] = forward_layer(im, w1, b1);
		[A2, Z2] = forward_layer(A1, w2, b2);
		[A3, Z3] = forward_layer(A2, w3, b3);
		
		% Compute Mean Squared Cost: - Sums over batch samples by itself
		traCost(epoch) = mean((A3-lbls').^2);
		
		% Compute Mean Classification Error: (Between 0 and 1)
		traPred(A3 < 0) = -1; traPred(A3 >= 0) = 1;
		traError(epoch) = sum(lbls' ~= traPred) ./ batch_size;
		
		% Backward Pass:
		% Local gradient for output layer:
		del3 = (A3-lbls');
		[dCost_dw3, dCost_db3, del2] = backprop_layer(Z3, A2, del3, w3);
		[dCost_dw2, dCost_db2, del1] = backprop_layer(Z2, A1, del2, w2);
		[dCost_dw1, dCost_db1, ~] = backprop_layer(Z1, im, del1, w1);
		
		% Delta Rule Update:
		w3 = w3 - lr * dCost_dw3;
		w2 = w2 - lr * dCost_dw2;
		w1 = w1 - lr * dCost_dw1;
		b3 = b3 - lr * dCost_db3;
		b2 = b2 - lr * dCost_db2;
		b1 = b1 - lr * dCost_db1;
	end
	
	%% After an epoch is done, calculate error on test set:
	% We need cross validation, using test like this is not ideal in my
	% opinion. However as the question asked it this way, I will do this
	% this way. Note that test set will be given the same batch size;
	% this is made for the equations to hold:
	for b = 1:size(testims,3)
		test_im = testims(:,:,b);
		test_lbls = testlbls(:,b);
		
		% Forward Pass:
		[val_A1, val_Z1] = forward_layer(test_im, w1, b1);
		[val_A2, val_Z2] = forward_layer(val_A1, w2, b2);
		[val_A3, val_Z3] = forward_layer(val_A2, w3, b3);
		
		val_preds(:,b) = val_A3;
		
		% Check costs:
		testCost(epoch) = mean((val_preds(:,b)-test_lbls).^2);
		
		% Pick classes:
		val_preds(val_preds >= 0) = 1;
		val_preds(val_preds < 0) = -1;
		
		testError(epoch) = sum(val_preds(:,b) ~= test_lbls) ./ ...
			batch_size;
	end
	
	% Flatten predictions: (as computing is now complete)
	testPred = reshape(val_preds, 1, size(val_preds,1) * ...
		size(val_preds,2));
	testFlat = reshape(testlbls, size(testPred,1), ...
		size(testPred,2));
	
% 	% Check test error should not increase
% 	if epoch >= 2 && (abs(testCost(epoch) - testCost(epoch-1)) ...
% 			./ abs(testCost(epoch-1))) < 0.001
% 		break;
% 	end
	end
end
function [traCost, traError, testCost, testError, ...
	testPred, testFlat] = ...
	two_layer_model_momentum(trainims, trainlbls, testims, testlbls, ...
	w1, w2, w3, b1, b2, b3, ...
	max_epoch, batch_size, lr, mr)

%% Pre generate batches - For train and validation sets: ******************
% Ignores non-full batch (Some data lost but not much, easier to implement)
traSetSize = length(trainlbls);
valSetSize = length(testlbls);

traExcess = mod(traSetSize,batch_size);
valExcess = mod(valSetSize,batch_size);

trainims = trainims(:,1:end-traExcess);
trainlbls = trainlbls(1:end-traExcess);

% Reshape layer:
trainims = reshape(trainims, 1024, batch_size, ...
	length(trainlbls)./batch_size);
trainlbls = reshape(trainlbls, batch_size, ...
	length(trainlbls)./batch_size);

testims = testims(:,1:end-valExcess);
testlbls = testlbls(1:end-valExcess);

% Reshape layer:
testims = reshape(testims, 1024, batch_size, ...
	length(testlbls)/batch_size);
testlbls = reshape(testlbls, batch_size, ...
	length(testlbls)/batch_size);
% *************************************************************************

for epoch = 1:max_epoch
	% Keep w,b in temp
	wv3 = 0; wv2 = 0; wv1 = 0; bv3 = 0; bv2 = 0; bv1 = 0;
	% Goes over entire dataset -> go over all batches
	for batch = 1: size(trainims,3)
		% Set 1 batch: - go through all images from start to end
		% Every item in the batch gets added to weight update automatically
		% thanks to the matrix operations
		im = trainims(:,:,batch); % This will be 1024x{batchSize}
		lbls = trainlbls(:,batch); % this will be {batchSize}x1
		
		% Forward Pass:
		[A1, Z1] = forward_layer(im, w1, b1);
		[A2, Z2] = forward_layer(A1, w2, b2);
		[A3, Z3] = forward_layer(A2, w3, b3);
		
		% Compute Mean Squared Cost: - Sums over batch samples by itself
		traCost(epoch) = mean((A3-lbls').^2);
		
		% Compute Mean Classification Error: (Between 0 and 1)
		traPred(A3 < 0) = -1; traPred(A3 >= 0) = 1;
		traError(epoch) = sum(lbls' ~= traPred) ./ batch_size;
		
		% Backward Pass:
		% Local gradient for output layer:
		del3 = 2*(A3-lbls');
		[dCost_dw3, dCost_db3, del2] = backprop_layer(Z3, A2, del3, w3);
		[dCost_dw2, dCost_db2, del1] = backprop_layer(Z2, A1, del2, w2);
		[dCost_dw1, dCost_db1, ~] = backprop_layer(Z1, im, del1, w1);
		
		wv3 = mr * wv3 + lr * dCost_dw3; w3 = w3 - wv3;
		wv2 = mr * wv2 + lr * dCost_dw2; w2 = w2 - wv2;
		wv1 = mr * wv1 + lr * dCost_dw1; w1 = w1 - wv1;
		
		bv3 = mr * bv3 + lr * dCost_db3; b3 = b3 - bv3;
		bv2 = mr * bv2 + lr * dCost_db2; b2 = b2 - bv2;
		bv1 = mr * bv1 + lr * dCost_db1; b1 = b1 - bv1;
		
	end
	
	%% After an epoch is done, calculate error on test set:
	% We need cross validation, using test like this is not ideal in my
	% opinion. However as the question asked it this way, I will do this
	% this way. Note that test set will be given the same batch size;
	% this is made for the equations to hold:
	for b = 1:size(testims,3)
		test_im = testims(:,:,b);
		test_lbls = testlbls(:,b);
		
		% Forward Pass:
		[val_A1, val_Z1] = forward_layer(test_im, w1, b1);
		[val_A2, val_Z2] = forward_layer(val_A1, w2, b2);
		[val_A3, val_Z3] = forward_layer(val_A2, w3, b3);
		
		val_preds(:,b) = val_A3;
		
		% Check costs:
		testCost(epoch) = mean((val_preds(:,b)-test_lbls).^2);
		
		% Pick classes:
		val_preds(val_preds >= 0) = 1;
		val_preds(val_preds < 0) = -1;
		
		testError(epoch) = sum(val_preds(:,b) ~= test_lbls) ./ ...
			batch_size;
	end
	
	% Flatten predictions: (as computing is now complete)
	testPred = reshape(val_preds, 1, size(val_preds,1) * ...
		size(val_preds,2));
	testFlat = reshape(testlbls, size(testPred,1), ...
		size(testPred,2));
	
% 	% Check test error should not increase
% 	if epoch >= 2 && (abs(testCost(epoch) - testCost(epoch-1)) ...
% 			/ abs(testCost(epoch-1))) < 0.001
% 		break;
% 	end
	
end
end

% Question 3 Helpers:
function [J,dJ]=cross_entropy(y,a)
logPred = log(a);
minLogPred = log(1-a);

logPred(isinf(logPred)) = 0; % Convert infs to 0, stabilize X-entropy

J= -sum(sum((y .* logPred + (1-y) .* minLogPred)))/200;
dJ=a-y;
end
function [dz] = grad_sigmoid(z)
% Gradient of the sigmoid function with respect to variable z.
dz = sigmoid(z) .* (1-sigmoid(z));
end
% function [dZ] = grad_softmax(Z)
% dZ = softmax(Z) .* (1-softmax(Z));
% end
function [htowStar, htobStar, ethwStar, ethbStar, wewStar, epTr, epVal] = ...
	model(trainx, traind, valx, vald, htow, htob, ethw, ethb, wew, ...
	max_epoch, mr, lr)

% Hard coded early stopping values
minValErr = Inf; count = 0; tolerance = 5;

% For test set, keep optimal values:
htowStar = zeros(size(htow)); htobStar = zeros(size(htob));
wewStar = zeros(size(wew));
ethwStar = zeros(size(ethw)); ethbStar = zeros(size(ethb));

tethw = zeros(size(ethw)); tethb = zeros(size(ethb));
thtow = zeros(size(htow)); thtob = zeros(size(htob));
twew = zeros(size(wew));
for ep = 1:max_epoch
	disp(['Epoch: ',num2str(ep)]);
	%% Training Phase:
	for its = 1:1862
		
		%% Embedding layer:
		batchWords = trainx(:,:,:,its); % 3x250x200
		batchWords = reshape(batchWords, 750, []); % 750x200
		embedBatchWords = wew' * batchWords; % 96x200
		
		%% Forward Propagation: (L1 then L2)
		zETH = ethw' * embedBatchWords + repmat(ethb,1,200); % 256x200
		aETH = sigmoid(zETH);
		
		zHTO =  htow' * aETH + repmat(htob,1,200); % 250x200
		aHTO = softmax(zHTO);
		
		%% Cost calculation:
		[traCost, dTrCost] = cross_entropy(traind(:,:,its),aHTO);
		batchCost(its) = (traCost);
		
		%% Backpropagation: (L2 then L1)
		dHTOb = mean(dTrCost,2); % For all 200 batches seperately.
		dHTOw =  0.005 .* (dTrCost * aETH');
		htoErr = (htow * dTrCost) .* grad_sigmoid(zETH);
		
		dETHb = mean(htoErr,2); % again for all 200 batches seperately.
		dETHw = 0.005 .* (htoErr * embedBatchWords');
		ethErr = (ethw * htoErr) .* embedBatchWords;
		
		% Now propagate back to embedding weights, linear, no extra delta
		% error applies here.
		dWEW = (batchWords * ethErr');
		clear embedBatchWords; % get ready for next reshape, not for debug
		
		%% Updates: w/ momentum:
		tethw = mr * tethw + lr * dETHw'; ethw = ethw - tethw;
		tethb = mr * tethb + lr * dETHb; ethb = ethb - tethb;
		thtow = mr * thtow + lr * dHTOw'; htow = htow - thtow;
		thtob = mr * thtob + lr * dHTOb; htob = htob - thtob;
		twew = mr * twew + lr * dWEW; wew = wew - twew;
		
	end
	disp(['Epoch Training Loss: ',num2str(mean(batchCost))]);
	epTr(ep) = mean(batchCost);
	
	%% Epoch finished: validation phase: - Predict
	
	% Propagate forward to Predict
	for batch = 1:232
		
		%% Embedding layer:
		valWords = valx(:,:,:,batch); % 3x250x200
		valWords = reshape(valWords, 750, []); % 750x200
		valBWords = wew' * valWords; % 96x200
		
		%% Forward Propagation: (L1 then L2)
		zETH = ethw' * valBWords + repmat(ethb,1,200); % 256x200
		aETH = sigmoid(zETH);
		
		zHTO =  htow' * aETH + repmat(htob,1,200); % 250x200
		aHTO = softmax(zHTO);
		
		%% Cost calculation:
		[valCost, ~] = cross_entropy(vald(:,:,batch),aHTO);
		batchValCost(batch) = valCost;
		clear valBWords; % get ready for next reshape, not for debug
		
	end
	disp(['Epoch Validation Loss: ',num2str(mean(batchValCost))]);
	epVal(ep) = mean(batchValCost);
	
	%% Do early stopping wrt validation loss
	if valCost<minValErr
		minValErr=valCost;
		count=0;
		%Best weights stored, even though I continue with updates
		wewStar = wew; ethwStar=ethw; ethbStar=ethb;
		htowStar=htow; htobStar=htob;
	else
		count=count+1;
	end
	if count == tolerance
		break; %stop early
	end
end

end
function [words_enc] = onehot_encode(words, vocab_size)
% Encode words to one hot vectors of vocabulary size.
% If words size is 1, just encodes the word on the go,
% Otherwise it one-hot encodes entire set of words given to it.
words_enc = zeros(size(words,1), vocab_size, size(words,2));
% For every sample, do the encoding
for j = 1:size(words,2)
	% do the encoding for every word 
	for i=1:size(words,1)
		words_enc(i,words(i,j),j) = 1;
	end
end
end
function [out] = sigmoid(z)
% Calculates sigmoid activation for the neurons, z: activation signal
out = 1./(1+exp(-z));
end
function [out] = softmax(o)
% Calculates the softmax for all output neurons given by output vector 'o'.
exp_o=exp(o);
out=exp_o./sum(exp_o,1);
end
function [aHTO] = predict(sampx, wew1, ethw1, ethb1, htow1, htob1,m)

sampx = reshape(sampx, 750, []); % 750x200
a0 = wew1' * sampx; % 96x200

%% Forward Propagation: (L1 then L2)
zETH = ethw1' * a0 + repmat(ethb1,1,m); % 256x200
aETH = sigmoid(zETH);

zHTO =  htow1' * aETH + repmat(htob1,1,m); % 250x200
aHTO = softmax(zHTO);


end