%%%
%Statistical Computing for Scientists and Engineers
%Homework 1
%Fall 2018
%University of Notre Dame
%%%
function log_likelihood = log_likelihood_value(alpha,beta,y,n)
log_likelihood = 0;
N=size(n,1);
for i =1:N
    %Add code below
    log_likelihood1 = 
    %Add code above
    log_likelihood = log_likelihood+log_likelihood1;
    
end
end
