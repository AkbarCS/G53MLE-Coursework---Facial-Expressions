function [cm, actual_labels, predict_labels] = create_cm(predict,actual)
%CREATE_CM Summary of this function goes here
%   Detailed explanation goes here
    [row, col] = size(predict);
    predict_labels = zeros(1,col);
    actual_labels = zeros(1,col);
    for i = 1: col
        [predict_value, predict_argmax] = max(predict(:,i));
        [actual_value, actual_argmax] = max(actual(:,i));
        predict_labels(i) = predict_argmax;
        actual_labels(i) = actual_argmax;
    end
    
    cm = confusionmat(actual_labels,predict_labels);
    
    recall = zeros(1,row);
    precision = zeros(1,row);
    for i = 1: row
        recall(i) = cm(i,i)/ sum(cm(i,:));
        precision(i) = cm(i,i)/sum(cm(:,i));
    end
   


    
end

