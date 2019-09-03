%%%
%Statistical Computing for Scientists and Engineers
%Homework 1
%Fall 2018
%University of Notre Dame
%%%
clc
clear all
close all

A = exprnd(5,20,1);
mean_A = mean(A);
%Add code below
lambda_MLE = 
%Add code above
for alpha = 1:1:40
    beta = 100;
    n = length(A);
    %Add code below
    lambda_MAP = 
    %Add code above
    mse(alpha) = mean((0.2 - lambda_MAP).^2);
end
figure1 = figure;
axes1 = axes('Parent',figure1);
p1 = plot(mse,'k')
xlabel('\alpha','FontSize',15)
ylabel('MSE','FontSize',15)
set(p1,'LineWidth',2)
set(axes1,'FontSize',15,'FontWeight','bold');
saveas(figure1,'Solution_6b.png')
