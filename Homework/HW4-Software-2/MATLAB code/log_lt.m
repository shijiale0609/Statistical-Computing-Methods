
function y = log_lt(b,s2,X,y,N)
% likelihood function for normal data
    y = -N/2*log(s2)-5/2*sum(1+(X*b-y).^2/4/s2);
end