
%% loading data
clear;
clc;

fprintf('Loading data...\n')

load('Z:\G53MLE\data\data\assessment\regression headpose\headpose.mat');
load('Z:\G53MLE\data\data\assessment\regression headpose\facialPoints.mat');

label = pose(:, 6);

data = reshape(permute(points, [2, 1, 3]), [66 * 2, 8955]);
data = data';

%set up useful variables
[total_m, total_n]=size(data);         %num of instances and features
errors = zeros(1,10);
fold_inner = 3;
fold_outer = 10;
num_para = 6;
errors_inner = zeros(1, num_para);
opt = zeros(fold_inner * fold_outer, num_para);
svs = zeros(fold_outer * fold_inner, num_para);

%% 10-fold_CV
%random divide data into 10 packages
indices = crossvalind('Kfold', data(1:total_m, total_n),10);

for k = 1 : fold_outer
    
    %get indices for validation set and training set
    val_indices = (indices == k);
    train_indices = ~val_indices;
    
    %construct validation set and training set
    Xval = data(val_indices, :);
    Yval = label(val_indices);
    X = data(train_indices, :);
    Y = label(train_indices);
    
%%%%%%================> train and compute error
%     kernel = 'rbf';
%     sigma = 45;
%     q = 1;
%     e = 0.3;
%    
%     predict = trainSVM(kernel, sigma, q, e, X, Y, Xval);
%     plot_predict(Yval, predict);
%     errors(k) = computeRegCost(predict, Yval);
%     errors
%%%%%%=============================================
    

%%%%%%================> parameter arrays
%     kernels = ['linear', 'polynomial','rbf'];        % change manully inside the loop
%     sigmas = [45, 50, 55, 60, 65, 70];     % should be double
%     qs = [1, 2, 3, 4, 5, 6];               % should be positive int
      es = [0.1, 0.3, 0.7, 1.2, 1.5, 2.4];   % non-negative scalar value
%%%%%%=============================================

%%%%%%================> inner cross validation
     [p, q] = size(X); 
     indices1 = crossvalind('Kfold', X(1:p, q), fold_inner);
    
     for j = 1 : fold_inner
         
         test_indices_inner = (indices1 == j); 
         train_indices_inner = ~test_indices_inner;
         
         %Training set - 2 sets 
         data_train_inner = X(train_indices_inner,:); 
         label_train_inner = Y(train_indices_inner,:);
         
         %Testing set - 1 
         data_test_inner = X(test_indices_inner,:); 
         label_test_inner = Y(test_indices_inner,:);

         kernel = 'linear';      % <=== change manully here
         for count = 1 : num_para
             
             %=====chosen values=====   %<===comment out params being tuned
             sigma = 45;    % no need to tune sigma for polynomial kernel, use default 
             q = 1;         % no need to tune q for rbf kernel, use default       
             %e = 0.3;
             %=======================
             
             %======tune param=======   %<===comment out params NOT being tuned
             %sigma = sigmas(count);     % no need to tune sigma for polynomial kernel, use default 
             %q = qs(count);             % no need to tune q for rbf kernel, use default       
             e = es(count);
             %=======================
             
             [predict_inner,sv] = trainSVM(kernel, sigma, q, e, data_train_inner, label_train_inner, data_test_inner);
             svs(3 * (k - 1) + j, count) = sv
             opt(3 * (k - 1) + j, count) = computeRegCost(predict_inner, label_test_inner);
         end
         
     end
%%%%%%=============================================
end

%%%%%%================> print error for outer cv
% err = mean(errors);
% fprintf('CV error is: %f\n', err);
%=============================================

%%%%%%================> print error for inner cv
err = mean(opt);
fprintf('CV error is: %f\n', err);
%=============================================
%% NN
%predict = trainNN(X, Y, Xval)

%% performance evaluation
%err = computeRegCost(predict, Yval);
