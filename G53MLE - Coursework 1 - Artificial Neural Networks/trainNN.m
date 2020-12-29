function predict = trainNN(X, Y, Xval)

%initialise  NN
P = X';
T = Y';
S = [4,4,4,4];

NET = newff(P,T,S);

%train NN
NET.trainParam.show = 7;  
NET.trainParam.epochs = 250;
NET.trainParam.goal = 0.005;
NET.trainParam.max_fail = 10;

[NET, TR] = train(NET, X', Y');

%cv result of NN
predict = sim(NET, Xval');
predict = predict';

end