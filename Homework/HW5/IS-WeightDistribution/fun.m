% -------------------------------------------------------------------------
%
% Copyright: MPDC Laboratory, Cornell University, 2010 
% http://mpdc.mae.cornell.edu/ (Prof. N. Zabaras)
%
% This solver is used to show the distribution of weighted values
% correspoding to two IS distribution which have the same target double
% exponential distribution
% -------------------------------------------------------------------------

function result = fun(x)

if(x>=0)
    result = 0.0009*exp(-1.1*x);
else 
    result = 0.0009*exp(1.1*x);
end
