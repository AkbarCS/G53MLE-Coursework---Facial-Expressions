function plot_predict(Yval, predict)

figure;
scatter(Yval,predict)
hold on
x=-40:0.001:40;
y=x;
plot(x,y)

end