function [tree] = decision_tree_learning(features, labels)
%DECISION_TREE_LEARNING Summary of this function goes here
%   Detailed explanation goes here
% tree.op is label for node. tree.kids = [] / tree.class is label for
% returning class
    tree.op = [];
    tree.threshold = [];
    tree.kids = cell(1,2);
    tree.class = [];
    %[row, col] = size(features);
    if all(labels == labels(1))
        tree.class = labels(1);
        tree.kids =[];
    else
        % what is target?
        [best_feature, best_threshold] = choose_attribute(features, labels);
        tree.op = best_feature;
        tree.threshold = best_threshold;
        % tree.kids
        for i = 1:2
            if i == 1
                index = features(:, best_feature) <= best_threshold;
            else
                index = features(:, best_feature) > best_threshold;
            end
            examples = features(index,:);
            targets = labels(index,:);
            if isempty(examples)
                tree.class = majority_value(targets);
                tree.kids =[];
            else
                tree.kids{i} = decision_tree_learning(examples,targets);
            end
        end
    end 
end

