
%% loading data
fprintf('Loading data...\n')

load('Z:\G53MLE\data\data\assessment\regression headpose\facialPoints.mat');
load('Z:\G53MLE\data\data\assessment\regression headpose\headpose.mat');

label = pose(:, 6);

data = reshape(permute(points, [2, 1, 3]), [66 * 2, 8955]);
data = data';

%set up useful variables
[total_m, total_n] = size(data);         %num of instances and features
errors = zeros(1, 10);

%% 10-fold_CV
%random divide data into 10 packages
indices = crossvalind('Kfold', data(1:total_m, total_n),10);

for k = 1 : 10
    
    %get indices for validation set and training set
    val_indices = (indices == k);
    train_indices = ~val_indices;
    
    %construct validation set and training set
    Xval = data(val_indices, :);
    Yval = label(val_indices);
    X = data(train_indices, :);
    Y = label(train_indices);
    
    %train and compute error
    predict = trainNN(X, Y, Xval);
    errors(k) = computeRegCost(predict, Yval);
end

err = mean(errors);
fprintf('CV error is: %f\n', err);

%% NN
%predict = trainNN(X, Y, Xval)

%% performance evaluation
%err = computeRegCost(predict, Yval);
