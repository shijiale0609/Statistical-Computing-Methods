%%%
%Statistical Computing for Scientists and Engineers
%Homework 4
%Fall 2018
%University of Notre Dame
%%%
%accept_reject
%%%%%%%%%% add code below %%%%%%%%



%%%%%%%%%% add code above %%%%%%%%%%%%


%Plot the convergence
%%%%%%%%%%%% add code below %%%%%%%%%%%%


%%%%%%%%%%%%% add code above %%%%%%%%%%%%%%%%%%%%
figure1 = figure;
axes1 = axes('Parent',figure1);
hold(axes1,'on');
p1=plot(RMSE1,'b')
set(p1,'LineWidth',2)
set(axes1,'FontSize',15,'FontWeight','bold');
saveas(figure1,'Convergence_Problem-1a.png')

figure2 = figure;
p2 = plot(x,f(x),'k') % f(x) is the True distribution
hold all
hleg =legend('sampled histogram','True distribution')
leg = legend('show');
set(hleg,'Location','NorthEast')
set(hleg,'Interpreter','none')
set(p2,'LineWidth',2)
saveas(figure2,'Problem-1a.png')
