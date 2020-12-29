function [prediction,num_sv] = trainSVM(kernel, sigma, q, e, X, Y, Xval)

%train SVM

 if strcmp(kernel, 'rbf')
     fprintf('building rbf svm...\n')
     Mdl = fitrsvm(X, Y, 'KernelFunction',kernel, 'KernelScale', sigma, 'BoxConstraint', 1, 'Epsilon', 0.3);
 elseif strcmp(kernel, 'polynomial')
     fprintf('building polynomial svm...\n')
     Mdl = fitrsvm(X, Y, 'KernelFunction',kernel, 'PolynomialOrder', q, 'BoxConstraint', 1, 'Epsilon', 0.3);
 elseif strcmp(kernel, 'linear')
     fprintf('building linear svm...\n')
     Mdl = fitrsvm(X, Y, 'KernelFunction', kernel, 'Epsilon', e);
 else
     fprintf('building other kernel svm...\n')
     Mdl = fitrsvm(X, Y, 'KernelFunction', kernel, 'BoxConstraint', 1, 'Epsilon', 0.3);
 end

%Mdl = fitrsvm(X, Y, 'KernelFunction', 'linear');

%cv result of SVM
%[~,PS] = mapminmax(X);
%Xval=mapminmax(Xval);
sv = Mdl.SupportVectors;
num_sv = size(sv, 1);
prediction = predict(Mdl, Xval);
%predict = predict';

end