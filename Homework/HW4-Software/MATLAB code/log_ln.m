
function y = log_ln(b,s2,X,y,N)
% likelihood function for normal data
    y = -N/2*log(s2)-1/2/s2*sum((X*b-y).^2);
end