%% Problem 1

% Initialization
a = 5; b = a-1; c = 1/(2*a-1);
M = exp(-a+1)*(a-1)^(a-1)*pi*sqrt(2*a-1)/gamma(a);
Ns = 1e4;
xs = zeros(Ns,1);
i = 0;
% Accept-Reject method
while i < Ns
    xc = b+tan(pi*(rand-0.5))/sqrt(c); % transform rand sample to a Cauchy sample
    if xc > 0 % directly reject if out of Gamma dist support
        r = pi*(1+c*(xc-b)^2)*xc^(a-1)*exp(-xc)/(sqrt(c)*gamma(a)*M);
        if r > rand
            i = i + 1;
            xs(i) = xc;
        end
    end
end
% Plots
x = linspace(0,30,500)';
y1 = x.^(a-1).*exp(-x)/gamma(a);
y2 = M*sqrt(c)./(pi*(1+c*(x-b).^2));
figure()
plot(x,y1,'r',x,y2,'b','LineWidth',3);
legend('Gamma distribution','Upper envelope');
title('a = 5, M = 1.84');
figure()
[f,xh]=hist(xs,100);
bar(xh,f/trapz(xh,f));hold on
plot(x,y1,'r','LineWidth',3);
legend('Sample histogram','Exact distribution');



%% Problem 2

% Initilization
Ns = 1e4;
xs = 1.5;    % initial sample
p = gampdf(1.5,xs(end),1)*(sin(pi*xs(end))^2);
q = exppdf(xs,5);
% Metropolis-Hasting method
j = 0;
for i = 2:Ns
    xc = exprnd(5);   % lambda = 5;
    pt = gampdf(1.5,xc,1)*(sin(pi*xc)^2);
    qt = exppdf(xc,5);
    if rand<pt/p*(q/qt)
        xs = [xs;xc];
        p = pt; q = qt;
        j = j+1;
    else
        xs = [xs;xs(end)];
    end
end


xs(1:Ns/4) = [];    % burn-in discarded

% Plot of proposal and posterior distribution
x = linspace(0,10,500);
y1 = exppdf(x,5); % proposal
y2 = exp(-1.5)*1.5.^(x-1).*(sin(pi*x)).^2./gamma(x);
y2 = y2/trapz(x,y2);
figure()
plot(x,y1,'-g',x,y2,'-r','LineWidth',3);
legend('Proposal','Posterior');

figure()
[f,xh]=hist(xs,100);
bar(xh,f/trapz(xh,f));hold on
plot(x,y2,'r','LineWidth',3);
legend('Sample histogram','Exact distribution');

%% Problem 3

C=[1 -0.5; -0.5 1];
m=[1 1];

Ns = 1e4;

%% Problem 3a
xs = zeros(Ns,2);

for i=2:Ns
    % x1|x2 ~ N( mu(x1|x2), C(x1|x2) ). So, mu(x1|x2) = m(1) + C(1,2)*(x(2)-m(2))/C(2,2)
    % C(x1|x2) = C(1,1) - C(2,2)*C(1,2)^2
    xs(i,1) = m(1) + C(1,2)*(xs(i-1,2)-m(2))/C(2,2) + sqrt(C(1,1) - (C(1,2)^2)/C(2,2))*randn();
    xs(i,2) = m(2) + C(1,2)*(xs(i,1)-m(1))/C(1,1) + sqrt(C(2,2) - (C(1,2)^2)/C(1,1))*randn();
end


x = linspace(-2,4,500);
y = 1/sqrt(2*pi)*exp(-0.5*(x-1).^2);
[f,xh]=hist(xs(:,1),100);
figure()
bar(xh,f/trapz(xh,f));
hold on
plot(x,y,'r','LineWidth',3);
legend('Sample histogram','Exact distribution');


[f,xh]=hist(xs(:,2),100);
figure()
bar(xh,f/trapz(xh,f));
hold on
plot(x,y,'r','LineWidth',3);
legend('Sample histogram','Exact distribution');


x1 = -3:.05:5; x2 = -3:.05:5;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],m,C);
F = reshape(F,length(x2),length(x1));
figure(10)
subplot(2,2,1)
contour(x1,x2,F);
hold on
plot(xs(1:50,1),xs(1:50,2),'ro-','Linewidth',2);
xlim([-3,5]); ylim([-3,5]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Gibbs')

%% Problem 3b

xs = zeros(1,2);
p = mvnpdf(xs(end,:),m,C);

for i=2:Ns
    xt = mvnrnd(xs(end,:),eye(2));
    pt = mvnpdf(xt,m,C);
    if rand<pt/p
        xs = [xs;xt];
        p = pt;
    else
        xs = [xs;xs(end,:)];
    end
end


figure
subplot(1,2,1)
contour(x1,x2,F,'LineWidth',3);
xlim([-3,5]); ylim([-3,5]);
xlabel('x1'); ylabel('x2'); title('Probability Density');
subplot(1,2,2)
scatter(xs(:,1),xs(:,2),'.')
xlabel('x1'); ylabel('x2'); title('10000 Samples ');
xlim([-3,5]); ylim([-3,5]);

figure(10)
subplot(2,2,2)
contour(x1,x2,F);
hold on
plot(xs(1:50,1),xs(1:50,2),'ro-','Linewidth',2);
xlim([-3,5]); ylim([-3,5]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Block MH')



%% Problem 3c

xs = zeros(1,2);
p = mvnpdf(xs(end,:),m,C);
q1 = normpdf(xs(end,1));
q2 = normpdf(xs(end,2));

for i=2:Ns
    xt1 = randn;
    pt = mvnpdf([xt1 xs(end,2)],m,C);
    qt = normpdf(xt1);
    if rand<(pt/p)*(q1/qt)
        xs = [xs;[xt1 xs(end,2)]];
        q1 = qt; p = pt;
    else
        xs = [xs;xs(end,:)];
    end
    
    xt2 = randn;
    pt = mvnpdf([xs(end,1) xt2],m,C);
    qt = normpdf(xt2);
    if rand<(pt/p)*(q2/qt)
        xs(end,2) = xt2;
        q2 = qt; p = pt;
    end
end


figure
subplot(1,2,1)
contour(x1,x2,F,'LineWidth',3);
xlim([-3,5]); ylim([-3,5]);
xlabel('x1'); ylabel('x2'); title('Probability Density');
subplot(1,2,2)
scatter(xs(:,1),xs(:,2),'.')
xlabel('x1'); ylabel('x2'); title('10000 Samples ');
xlim([-3,5]); ylim([-3,5]);

figure(10)
subplot(2,2,3)
contour(x1,x2,F);
hold on
plot(xs(1:50,1),xs(1:50,2),'ro-','Linewidth',2);
xlim([-3,5]); ylim([-3,5]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Component-wise MH')



%% Problem 3d
 

e = 0.1;
L = 20;

xs = zeros(1,2);


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



figure
subplot(1,2,1)
contour(x1,x2,F,'LineWidth',3);
xlabel('x1'); ylabel('x2'); title('Probability Density');
subplot(1,2,2)
scatter(xs(:,1),xs(:,2),'.')
xlabel('x1'); ylabel('x2'); title('10000 Samples with L = 20, \epsilon = 0.1');



figure(10)
subplot(2,2,4)
contour(x1,x2,F);
hold on
plot(xs(1:50,1),xs(1:50,2),'ro-','Linewidth',2);
xlim([-3,5]); ylim([-3,5]);
legend({'Samples','1st 50 States'},'Location','Northwest')
title('Hamiltonian MH')



%% Problem 4
% Load data
x = [4 4 7 7 8 9 10 10 10 11 11 12 12 12 12 13 13 13 13 14 14 14 14 15 15 15 16 16 17 17 17 18 18 18 18 19 19 19 20 20 20 20 20 22 23 24 24 24 24 25]';
y = [2 10 4 22 16 10 18 26 34 17 28 14 20 24 28 26 34 34 46 26 36 60 80 20 26 54 32 40 32 40 50 42 56 76 84 36 46 68 32 48 52 56 64 66 54 70 92 93 120 85]';


X = [ones(50,1),x,x.^2];    % regression matrix
b  = (X'*X)\X'*y;   % mle for beta   
s2 = mean((X/(X'*X)*X'*y-y).^2);    % mle for sigma2
vb = s2*inv(X'*X);  % covariance for beta
N = 50; % number of data points
Ns = 2e4;   % samples


% part a

bs2 = [b;1/s2];   % initial point for markov chain, here we set it to be the MLE point
rej = 0;    % accept/reject counter 
l = log_ln(b,s2,X,y,N); % initialize likelihood/proposal
p = mvnpdf(b,b,(vb))*gampdf(1/s2,(N-3)/2,2/N/s2);

for i = 2:Ns
    % new sample
    bt = zeros(4,1);
    bt(1:3) = mvnrnd(b,diag(diag(vb)),1);
    bt(1:3) = mvnrnd(b,vb,1);

    bt(4) = gamrnd((N-3)/2,2/N/s2);
    % proposal density and likelihood
    lt = log_ln(bt(1:3),1/bt(4),X,y,N);
    pt = mvnpdf(bt(1:3),b,vb)*gampdf(bt(4),(N-3)/2,2/N/s2);

    % compare to the previous sample
    if (unifrnd(0,1)<exp(lt-l)*(p/pt))
      bs2 = [bs2 bt];  % accept
      p = pt;
      l = lt;
    else
        if i>Ns/4   % if after burning period, count the times of rejection
            rej = rej+1;
        end
        bs2 = [bs2 bs2(:,end)]; % reject, keep previous
    end
end

bs2(:,1:Ns/4) = [];
Ns = Ns - Ns/4;


figure
subplot(4,1,1)
[f,xh]=hist(bs2(1,:),100);
bar(xh,f/trapz(xh,f));
subplot(4,1,2)
[f,xh]=hist(bs2(2,:),100);
bar(xh,f/trapz(xh,f));
subplot(4,1,3)
[f,xh]=hist(bs2(3,:),100);
bar(xh,f/trapz(xh,f));
subplot(4,1,4)
[f,xh]=hist(bs2(4,:),100);
bar(xh,f/trapz(xh,f));




figure
subplot(4,1,1)
plot(1:Ns,cumsum(bs2(1,:))./(1:Ns))
subplot(4,1,2)
plot(1:Ns,cumsum(bs2(2,:))./(1:Ns))
subplot(4,1,3)
plot(1:Ns,cumsum(bs2(3,:))./(1:Ns))
subplot(4,1,4)
plot(1:Ns,cumsum(bs2(4,:))./(1:Ns))

% par b
bs2 = [b;1/s2];   % initial point for markov chain, here we set it to be the MLE point
rej = 0;    % accept/reject counter 
l = log_lt(b,s2,X,y,N); % initialize likelihood/proposal
vbt = 2*vb;
p = mvnpdf(b,b,vbt)*gampdf(1/s2,(N-3)/2,2/N/s2);

for i = 2:Ns
    % new sample
    bt = zeros(4,1);
    bt(1:3) = mvnrnd(b,vbt,1);
    bt(4) = gamrnd((N-3)/2,2/N/s2);
    % proposal density and likelihood
    lt = log_lt(bt(1:3),1/bt(4),X,y,N);
    pt = mvnpdf(bt(1:3),b,vbt)*gampdf(bt(4),(N-3)/2,2/N/s2);

    % compare to the previous sample
    if (unifrnd(0,1)<exp(lt-l)*(p/pt))
      bs2 = [bs2 bt];  % accept
      p = pt;
      l = lt;
    else
        if i>Ns/4   % if after burning period, count the times of rejection
            rej = rej+1;
        end
        bs2 = [bs2 bs2(:,end)]; % reject, keep previous
    end
end

bs2(:,1:Ns/4) = [];

Ns = Ns - Ns/4;

figure
subplot(2,2,1)
[f,xh]=hist(bs2(1,:),100);
bar(xh,f/trapz(xh,f));
subplot(2,2,2)
[f,xh]=hist(bs2(2,:),100);
bar(xh,f/trapz(xh,f));
subplot(2,2,3)
[f,xh]=hist(bs2(3,:),100);
bar(xh,f/trapz(xh,f));
subplot(2,2,4)
[f,xh]=hist(bs2(4,:),100);
bar(xh,f/trapz(xh,f));



figure
subplot(2,2,1)
plot(1:Ns,cumsum(bs2(1,:))./(1:Ns))
subplot(2,2,2)
plot(1:Ns,cumsum(bs2(2,:))./(1:Ns))
subplot(2,2,3)
plot(1:Ns,cumsum(bs2(3,:))./(1:Ns))
subplot(2,2,4)
plot(1:Ns,cumsum(bs2(4,:))./(1:Ns))


