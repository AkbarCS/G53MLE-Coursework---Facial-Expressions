%% loading data
clear
clc

fprintf('Loading data...\n')

% Load data
load('Z:\G53MLE\data\data\assessment\binary smile\facialPoints.mat');
load('Z:\G53MLE\data\data\assessment\binary smile\labels.mat');

% Reshape data
points = permute(points, [2,1,3]);
points = reshape(points, [66 * 2, 150]);
points = points';

%% CHOOSE-ATTRIBUTE
%[best_feature, best_threshold] = choose_attribute(points, labels)
tree = decision_tree_learning(points, labels);
DrawDecisionTree(tree)
output = zeros(150,1);
for i = 1:150
    output(i,1) = tree_output(tree,points(i,:));
end
