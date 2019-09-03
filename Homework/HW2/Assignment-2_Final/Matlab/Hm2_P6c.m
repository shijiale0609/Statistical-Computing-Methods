%%%
%Statistical Computing for Scientists and Engineers
%Homework 1
%Fall 2018
%University of Notre Dame
%%%
clc
clear all
close all


for i = 1:1:500
    A = exprnd(5,i,1);
    mean_A = mean(A);
    %Add code below
    lambda_MLE =
    %Add code above
    mse_MLE(i) = mean((0.2 - lambda_MLE).^2);
    alpha = 30
    beta = 100;
    n = length(A);
    %Add code below
    lambda_MAP = 
    %Add code above
    mse_MAP(i) = mean((0.2 - lambda_MAP).^2);
end
figure1 = figure;
axes1 = axes('Parent',figure1);
p1 = plot(mse_MLE,'k')
hold all
p2 = plot(mse_MAP,'b')
hleg =legend('MLE','MAP')
leg = legend('show');
set(p1,'LineWidth',2)
set(p2,'LineWidth',2)
set(hleg,'Location','NorthEast')
set(hleg,'Interpreter','none')

xlabel('\alpha','FontSize',15)
ylabel('MSE','FontSize',15)
set(axes1,'FontSize',15,'FontWeight','bold');
saveas(figure1,'Solution_6c.png')
