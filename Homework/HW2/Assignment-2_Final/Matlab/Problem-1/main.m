%%%
%Statistical Computing for Scientists and Engineers
%Homework 1
%Fall 2018
%University of Notre Dame
%%%
clc
clear all
close all
data = importdata('data.csv');
t = data(:,1);
n = data(:,2);
x_size=100;
y_size=400;
X = linspace(-2.3,-1.3,x_size);
Y = linspace(1,5,y_size);
for i = 1:x_size
    for j = 1:y_size
        x=X(i);
        y=Y(j);     
beta(i,j) = beta_value(x,y);
alpha(i,j) = alpha_value(x,y);
prior(i,j) = log_prior_value(alpha(i,j),beta(i,j));
likelihood(i,j) = log_likelihood_value(alpha(i,j),beta(i,j),t,n);
posterior(i,j) = prior(i,j) +likelihood(i,j);
jacobian(i,j) = log(alpha(i,j))+log(beta(i,j));
transformed(i,j) = posterior(i,j) +jacobian(i,j);
    end 
end
expo = exp(transformed-max(transformed(:)));
contourf(X,Y,transpose(expo));
xlabel('$\log(\alpha/\beta)$','Interpreter','latex')
ylabel('$\log(\alpha+\beta)$','Interpreter','latex')
