% *************************************************************************
% ECS797P - Machine Learning for Visual Data Analytics
% Assignment 3 - Facial Age Estimation by (Linear) Regression
% Hasan Emre Erdemoglu - 200377106, Due: April 1st 2021
% Acknowledgements: *******************************************************
% libsvm library is used in this assignment.
% LibSVM library:      http://www.csie.ntu.edu.tw/~cjlin/libsvm/
addpath(genpath('libsvm')); % add the path of code to your workspace
% libsvm has to be compiled via this comment when necessary:
% run('/libsvm/matlab/make')
% *************************************************************************
%% Part 1: Getting Started -- Settings:
clear; clc; % For debugging purposes
database_path = './data_age.mat';

% Cumulative error level setting: (Global)
err_level = 5;

%% Part 2.1 - 2.2: Read train/test data and regress() #####################
load(database_path); % Part 2.1, loads the dataset.

xtrain = trData.feat; % Features pulled from training data struct
ytrain = trData.label; % Labels pulled from training data struct
w_lr = regress(ytrain,xtrain); % Use Linear Regr. from MATLAB fcns.
% #########################################################################

%% Part 2.3: Read testing data, apply learned linear regression ###########
xtest = teData.feat; % Features pulled from testing data struct
ytest = teData.label; % Labels pulled from testing data struct
yhat_test = xtest * w_lr; % Use learned weights from training to predict

% Plot graph to show the negative age predictions, and when they happen:
neg_preds_idx = yhat_test < 0; % Filter negative predictions
figure; stem(ytest(neg_preds_idx == 1)); hold on;
stem(yhat_test(neg_preds_idx == 1));
title('Ground Truth and MLR Predictions (when MLR Prediction is < 0)');
xlabel('Samples'); ylabel('Age/Estimated Age'); axis tight;
legend('Ground Truth', 'MLR Predictions');

figure; stem(ytest); hold on; stem(yhat_test);
title('Ground Truth and MLR Predictions (All Samples)');
xlabel('Samples'); ylabel('Age/Estimated Age'); axis tight;
legend('Ground Truth', 'MLR Predictions');
% #########################################################################

%% Part 2.4: Compute the MAE and CS value for linear regression ###########
disp('Part 2.4 Outputs:');
% Cumulative Error calculation: (From definition)
cs = sum(abs(ytest-yhat_test) <= err_level)/size(ytest,1) * 100;
fprintf('Cumulative Error with %d levels is %f.\n',err_level,cs);

% Mean Absolute Error calculation: (From definition)
mae = sum(abs(ytest-yhat_test))/size(ytest,1);
fprintf('Mean Absolute Error is %f.\n',mae);
disp(' ');
% #########################################################################

%% Part 2.5: Generate CS vs Error Level Plot (Ranging from 1 to 15) #######
cs_5 = zeros(15,1);
for i = 1:size(cs_5,1)
    % Use CS calculation from definition, loop over different error levels
    cs_5(i) = sum(abs(ytest-yhat_test) <= i)/size(ytest,1) * 100;
end

% Print the plot:
figure; plot(1:15, cs_5, 'b--o'); grid on; legend('Linear Regresion');
title('Cumulative Score Against Cumulative Score Error Level');
xlabel('Cumulative Error Level'); ylabel('Cumulative Score'); axis tight;
% #########################################################################

%% Part 2.6.1: Use MATLAB built-in functions to do following: #############
disp('Part 2.6 Outputs:');
% Part 2.6.1: MAE and CS (level: 5) for Partial Least Squares Model
[~,~,~,~,~,PCTVAR] = plsregress(xtrain,ytrain,201);

% Do the test over all dimensions; find the value which gives best
% variance explaine with least loading parameters (From plot: 10 is good)
figure; plot(1:201,cumsum(100*PCTVAR(2,:)),'-bo'); axis tight; grid on;
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');
title('PLS Components vs Variance Explained');

% Retrain with 10 parameters, regress on test data:
[xl,yl,xs,ys,beta,PCTVAR] = plsregress(xtrain,ytrain,10);
yhat_test = [ones(size(xtest,1),1), xtest]*beta;

% Cumulative Error calculation: (From definition)
cs = sum(abs(ytest-yhat_test) <= err_level)/size(ytest,1) * 100;
fprintf('Cumulative Error for PLS with %d levels is %f.\n',err_level,cs);

% Mean Absolute Error calculation: (From definition)
mae = sum(abs(ytest-yhat_test))/size(ytest,1);
fprintf('Mean Absolute Error for PLS is %f.\n',mae);
disp(' ');

%% Part 2.6.2: MAE and CS (level: 5) for Regression Tree Model
% Fit the regression tree, optimize the hyperparameters:
tree = fitrtree(xtrain,ytrain,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName',...
    'expected-improvement-plus','Verbose',0));

% Predict test labes given tree:
yhat_test = predict(tree,xtest);

disp('');
% Cumulative Error calculation: (From definition)
cs = sum(abs(ytest-yhat_test) <= err_level)/size(ytest,1) * 100;
fprintf('Cumulative Error for Regression Tree with %d levels is %f.\n', ...
    err_level,cs);

% Mean Absolute Error calculation: (From definition)
mae = sum(abs(ytest-yhat_test))/size(ytest,1);
fprintf('Mean Absolute Error for Regression Tree is %f.\n',mae);
% #########################################################################

%% Part 2.7: MAE & CS value (level: 5) for SV Regression by using LIBSVM ##
% Selecting epsilon-SVR(s=3), kernel type, cost and gamma of kernel fcn is
% selected by 5-fold cross-validation.
bestcv=inf; bestc = 0; bestep = 0;
disp('Part 2.7 Outputs:');
% Using Linear Kernel (No search in Gamma values):
i = 0;
for log10c = -1:0.5:2 % Sensible range of C values (7 tests)
    for log10ep = -2:0.5:1 % Sensible range of epsilon values (7 tests)
        cmd = ['-s 3 -v 5 -t 0 -c ', num2str(10^log10c),' -p ', ...
            num2str(10^log10ep), ' -b 1 -q'];
        cv = svmtrain(ytrain, xtrain, cmd);
        if (cv <= bestcv)
            bestcv = cv; bestc = 10^log10c; bestep = 10^log10ep;
        end
        i=i+1;
        fprintf('Iteration %d. ', i);
        fprintf('Epsilon: %d, C: %d .\n', 10^log10ep, 10^log10c);
        fprintf( 'Best C: %f, Best Epsilon: %f. \n\n', bestc, bestep);
        disp('');
    end
end

% Set the options and retrain best found model:
options = sprintf('-s 3 -t 0 -c %f -p %f -q', bestc, bestep);
model=svmtrain(ytrain, xtrain, options);
[ytest_hat, ~ , ~] = svmpredict(ytest, xtest, model);

% Set the options and retrain best found model:
% options = sprintf('-s 3 -t %d -c %f -g %f -b 1 -q', 1, 1/201, 1);

% Compute MAE and CS (with 5 levels):
disp('Part 2.7 Outputs:');
% Cumulative Error calculation:
fprintf( 'Linear SVR --\nBest C: %f, Best Epsilon: %f. \n', bestc, bestep);
cs = sum(abs(ytest-ytest_hat) <= err_level)/size(ytest,1) * 100;
fprintf('Cumulative Error with %d levels is %f.\n',err_level,cs);

% Mean Absolute Error calculation:
mae = sum(abs(ytest-ytest_hat))/size(ytest,1);
fprintf('Mean Absolute Error is %f.\n',mae);
disp(' ');

%% Non-linear kernels:
% All the tests use 5-fold CV (-v 5), using silent mode (-q) and
% probability estimates (-b 1)
% Using non-linear kernels (Search space in kernel, gamma, c, epsilon):
i = 0; bestcv=inf; bestc = 0; bestg = 0; bestep = 0; bestk = 0;
for kernel = 1:3 % Remaining kernel types (3 tests)
    for log10c = -1:1:2 % Sensible range of C values (4 tests)
        for log10ep = -2:0.5:1 % Sensible range of epsilon values (7 tests)
            for log10g = -3:1:1 % Sensbl. range of gamma values (5 tests)
                cmd = ['-s 3 -v 5 -t ',kernel, ' -c ', num2str(10^log10c), ...
                    ' -p ',num2str(10^log10ep), ' -g ', log10g ' -b 1 -q'];
                cv = svmtrain(ytrain, xtrain, cmd);
                if (cv <= bestcv)
                    bestcv = cv; bestc = 10^log10c; bestk = kernel;
                    bestep = 10^log10ep; bestg = 10^log10g;
                end
                i=i+1;
                fprintf('Iteration %d. ', i);
                fprintf('Gamma: %d, kernel: %d. \n', 10^log10g, kernel);
                fprintf('Epsilon: %d, C: %d .\n', 10^log10ep, 10^log10c);
                fprintf( 'Best C: %f, Best Epsilon: %f. \n', bestc, bestep);
                fprintf('Best Gamma: %d, best kernel: %d. \nn', 10^log10g, bestk);
                disp('');
            end
        end
    end
end

options = sprintf('-s 3 -t %f -c %f -p %f -q -g %f', ...
    bestk, bestc, bestep, bestg);
model=svmtrain(ytrain, xtrain, options);
[ytest_hat, ~ , ~] = svmpredict(ytest, xtest, model);

% Compute MAE and CS (with 5 levels):
% Cumulative Error calculation:
fprintf( 'Non-Linear SVR --  Best Kernel, %d \n', bestk);
fprintf('Best C: %f, Best Epsilon: %f, Best Gamma: %f. \n', ...
    bestc, bestep, bestg);
cs = sum(abs(ytest-ytest_hat) <= err_level)/size(ytest,1) * 100;
fprintf('Cumulative Error with %d levels is %f.\n',err_level,cs);

% Mean Absolute Error calculation:
mae = sum(abs(ytest-ytest_hat))/size(ytest,1);
fprintf('Mean Absolute Error is %f.\n',mae);
disp(' ');
% #########################################################################