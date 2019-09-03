% -------------------------------------------------------------------------
%
% Copyright: MPDC Laboratory, Cornell University, 2010 
% http://mpdc.mae.cornell.edu/ (Prof. N. Zabaras)
%
% Generates n iid integers within 1~length(weight) according to 
% discrete probability weight
%
% This subroutine is applied to resampling in Sequential Monte Carlo
% methods 
%
% -------------------------------------------------------------------------

function y=polyrnd(weight, n)
[Y,ind]=sort(weight);
for i=1:n,
    u=rand();
    t=0;
    for k=1:length(Y),
        t=t+Y(k);
        if u<t,
            y(i)=ind(k);
            break;
        end
    end
end
