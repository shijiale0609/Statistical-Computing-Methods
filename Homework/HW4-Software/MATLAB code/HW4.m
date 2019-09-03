

%% Problem 3d
 

e = 0.1;
L = 20;

Ns = 10000;
m = [1 1];
C = [1 -0.5; -0.5 1];

xs = zeros(1,2);
x1=-3:.05:5;x2=-3:.05:5
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],m,C);
F = reshape(F,length(x2),length(x1));


for i=2:Ns
    U = 0.5*((xs(end,:)-m)*inv(C)*(xs(end,:)-m)');
    q = mvnrnd([0 0],eye(2));
    K = 0.5*(q*q');
    
    xt = xs(end,:);
    qt = q;
    for j = 1:L
        qt = (qt'-e*inv(C)*(xt-m)'/2)';
        xt = xt+e*qt;
        qt = (qt'-e*inv(C)*(xt-m)'/2)';
    end
    Ut = -0.5*((xt-m)*inv(C)*(xt-m)');
    Kt = -0.5*(qt*qt');
    
    if rand<exp(U+K-(Ut+Kt))
        xs = [xs;xt];
    else
        xs = [xs;xs(end,:)];
    end
   
end


x = linspace(-2,4,500);
y = 1/sqrt(2*pi)*exp(-0.5*(x-1).^2);

figure
subplot(1,2,1)
contour(x1,x2,F,'LineWidth',3);
xlabel('x1'); ylabel('x2'); title('Probability Density');
subplot(1,2,2)
scatter(xs(:,1),xs(:,2),'.')
xlabel('x1'); ylabel('x2'); title('10000 Samples with L = 20, \epsilon = 0.1');

[f,xh]=hist(xs(:,1),100);
figure
bar(xh,f/trapz(xh,f));
hold on
plot(x,y,'r','LineWidth',3);
legend('Sample histogram','Exact distribution');
title('p(x1|x2)')

figure
[f,xh]=hist(xs(:,2),100);
bar(xh,f/trapz(xh,f));
hold on
plot(x,y,'r','LineWidth',3);
legend('Sample histogram','Exact distribution');
title('p(x2|x1)')

figure
subplot(1,1,1)
contour(x1,x2,F);
hold on
plot(xs(1:50,1),xs(1:50,2),'ro-','Linewidth',2);
xlim([-3,5]); ylim([-3,5]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Hamiltonian MH')



