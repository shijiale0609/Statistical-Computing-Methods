% -------------------------------------------------------------------------
%
% Copyright: MPDC Laboratory, Cornell University, 2010 
% http://mpdc.mae.cornell.edu/ (Prof. N. Zabaras)
%
% This solver is used to show the distribution of weighted values
% correspoding to two IS distribution which have the same target double
% exponential distribution
% -------------------------------------------------------------------------

clear
close all;

xx=[];
w=[];
wt=[];
for i=1:500
    x=normrnd(0,1);
    if(x>3.2||x<-3.2)
        x=normrnd(0,1);
    end
    xx = [xx x];
    w=[w fun(x)/normpdf(x)];
    
end
plot(xx,w,'o');

hold on;
for i=1:length(w)
    plot([xx(i) xx(i)],[0 w(i)]);
end
xlabel('Value of x');
ylabel('Value of weights');
axis([-4 4 0 0.012]);

figure(2)
xx=[];
for i=1:800
    x=normrnd(0,10);
    xx = [xx x];
    wt=[wt fun(x)/tpdf(x,0.5)];    
end

plot(xx,wt,'o');
hold on;
for i=1:length(w)
    plot([xx(i) xx(i)],[0 wt(i)]);
end

axis([-30 30 -0.0001 0.0035]);
xlabel('Value of x');
ylabel('Value of weights');
