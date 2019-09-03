
clear;clc;
delta = 1;
LB = -1-delta;
UB = 1+delta;
N = 1E6; Pd = delta; M = 1/(2+2*delta);
Weight = zeros(N,1); x0 = 0;
for i = 1:N
    y = LB + (UB-LB)*rand();
    if (y > -1 && y < 1)
        Weight(i) = 0.5*(y - x0)/M;
        x = y;
    else
        Weight(i) = x0/Pd;
        continue;
    end
    
    while y > -1 && y < 1
        z = y;
        y = LB + (UB-LB)*rand();
        if (y > -1 && y < 1)
            Weight(i) = Weight(i)*0.5*(y - x)/M;
            x = y;
        end
    end
    Weight(i) = Weight(i)*z/Pd;
end
Est_fx = mean(Weight) % Estimated Value of f(x) at x = x0