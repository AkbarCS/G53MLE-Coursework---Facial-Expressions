function err = computeRegCost(predict, Yval)

%set up useful variables
m = length(Yval);  %num of instances

%calculate regression err
err = sqrt(sum((predict - Yval) .^ 2) / (2 * m));

end