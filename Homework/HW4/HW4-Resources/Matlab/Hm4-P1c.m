%%%
%Statistical Computing for Scientists and Engineers
%Homework 4
%Fall 2018
%University of Notre Dame
%%%
%accept_reject
%%%%%%%%%% add code below %%%%%%%%



%%%%%%%%%% add code above %%%%%%%%%%%%

figure2 = figure;
p2 = plot(x,f(x),'k') % f(x) is the True distribution
hold all
hleg =legend('sampled histogram','True distribution')
leg = legend('show');
set(hleg,'Location','NorthEast')
set(hleg,'Interpreter','none')
set(p2,'LineWidth',2)
saveas(figure2,'Problem-1a.png')
