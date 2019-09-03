%% Prob 4(e)
close all;
clear;clc;
T = 2000;num_particle = 100;
A = 0.7; C = 0.5; Q = 0.1; R = 0.1;
B = sqrt(Q); D = sqrt(R);
X = zeros(T,1); Y = zeros(T,1);

X(1) = normrnd(0,1);
Y(1) = normrnd(C*X(1), sqrt(R));
for i = 2:T
   X(i) = normrnd(A*X(i-1), sqrt(Q));
   Y(i) = normrnd(C*X(i), sqrt(R));
end
xx_estimate=zeros(T, num_particle);   % samples of the Markov states
weight=ones(size(xx_estimate));               % importance weights

% Initialization (time step = 1)
x0 = randn(1,num_particle);
sigma = 1/sqrt(1/B^2+C^2/D^2)*ones(1,num_particle);
mu = (sigma.^2).*(A*x0/B^2+C*Y(1)/D^2);
xx_estimate(1,:) = normrnd(mu,sigma,1,num_particle); 
x_origin = xx_estimate;
log_weight(1,:) = log(normpdf(Y(1),C*xx_estimate(1,:),D)) ...
    -log(normpdf(xx_estimate(1,:),mu,sigma)) ...
    +log(normpdf(xx_estimate(1,:),A*x0,B));
lkhmax = max(log_weight(1,:));
weight(1,:) = exp(log_weight(1,:)-lkhmax);
norm_weight(1,:) = weight(1,:)/sum(weight(1,:)); % normalized weight
ess(1) = 1/sum(norm_weight(1,:).^2);             % effective sample size
x_current = xx_estimate;
% save('samples1.mat','x_estimate');

% SMC estimation
for k = 2:T
    sigma = 1/sqrt(1/B^2+C^2/D^2)*ones(1,num_particle);
    mu = (sigma.^2).*(A*xx_estimate(k-1,:)/B^2+C*Y(k)/D^2);
    xx_estimate(k,:) = normrnd(mu,sigma,1,num_particle);
    x_current(k,:) = xx_estimate(k,:);
    log_weight(k,:) = log_weight(k-1,:)+ ...
        log(normpdf(Y(k),C*xx_estimate(k,:),D))- ...
        log(normpdf(xx_estimate(k,:),mu,sigma)) ...
        +log(normpdf(xx_estimate(k,:),A*xx_estimate(k-1,:),B)); 
    lkhmax = max(log_weight(k,:));
    weight(k,:) = exp(log_weight(k,:)-lkhmax);
    norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
    ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
%     cv(k) = sqrt(sum((num_particle*norm_weight(k,:)-1).^2)/num_particle); % coefficient of variance
    
    % Resampling procedure
    x_origin(k,:) = xx_estimate(k,:);
    % make resampling at each time step
    if ess < 1.0*num_particle,
        ind = resampleMultinomial(norm_weight(k,:));
        xx_estimate = xx_estimate(:,ind);         
        x_current(k,:) = xx_estimate(k,:);
        norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
        log_weight(k,:) = log(norm_weight(k,:));
    end
%    save(strcat('samples',num2str(k),'.mat'),'x_estimate');
end

% Plot the evolution of particles and corresponding marginal distributions
figure; 
% Plot the particles {X_{1:k}} at each time step k
subplot(2,1,2);
for i=1:num_particle
    plot(1:T,xx_estimate(1:T,i));
    hold on;
end
axis([T-50 T -3 2]);
grid on;title('Prob4(e)');
xlabel('time index'); ylabel('state');

% Plot the marginal distribution of p(x_k|y_{1:k}) from the samples
% {X_k^(i)}. The variable x_current records the samples {X_k} at each time
% step and can be used to give the empirical distribution of p(x_k|y_{1:k})

subplot(2,1,1);
XX = -3:0.1:2;
for i=1:T
    ff = ksdensity(x_current(i,:),XX,'width',0.5);
    plot(ff+(i-1),XX);
    hold on;
    sample_den(i) = interp1(XX,ff,X(i));
    plot(sample_den(i)+(i-1),X(i),'ro');
    hold on;
end
axis([T-50 T -3 2]);title('Prob4(e)');
xlabel('time index'); ylabel('state');

%% Prob_4(f)

xx_estimate=zeros(T, num_particle);   % samples of the Markov states
weight=ones(size(xx_estimate));               % importance weights

% Initialization (time step = 1)
x0 = randn(1,num_particle);
sigma = 1/sqrt(1/B^2+C^2/D^2)*ones(1,num_particle);
mu = (sigma.^2).*(A*x0/B^2+C*Y(1)/D^2);
xx_estimate(1,:) = normrnd(mu,sigma,1,num_particle); 
x_origin = xx_estimate;
log_weight(1,:) = log(normpdf(Y(1),C*xx_estimate(1,:),D)) ...
    -log(normpdf(xx_estimate(1,:),mu,sigma)) ...
    +log(normpdf(xx_estimate(1,:),A*x0,B));
lkhmax = max(log_weight(1,:));
weight(1,:) = exp(log_weight(1,:)-lkhmax);
norm_weight(1,:) = weight(1,:)/sum(weight(1,:)); % normalized weight
ess(1) = 1/sum(norm_weight(1,:).^2);             % effective sample size
x_current = xx_estimate;
% save('samples1.mat','x_estimate');

% SMC estimation
for k = 2:T
    sigma = 1/sqrt(1/B^2+C^2/D^2)*ones(1,num_particle);
    mu = (sigma.^2).*(A*xx_estimate(k-1,:)/B^2+C*Y(k)/D^2);
    xx_estimate(k,:) = normrnd(mu,sigma,1,num_particle);
    x_current(k,:) = xx_estimate(k,:);
    log_weight(k,:) = log_weight(k-1,:)+ ...
        log(normpdf(Y(k),C*xx_estimate(k,:),D))- ...
        log(normpdf(xx_estimate(k,:),mu,sigma)) ...
        +log(normpdf(xx_estimate(k,:),A*xx_estimate(k-1,:),B)); 
    lkhmax = max(log_weight(k,:));
    weight(k,:) = exp(log_weight(k,:)-lkhmax);
    norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
    ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
%     cv(k) = sqrt(sum((num_particle*norm_weight(k,:)-1).^2)/num_particle); % coefficient of variance
    
    % Resampling procedure
    x_origin(k,:) = xx_estimate(k,:);
    % make resampling at each time step
    if ess < 1.0*num_particle,
        ind = resampleSystematic(norm_weight(k,:));
        xx_estimate = xx_estimate(:,ind);         
        x_current(k,:) = xx_estimate(k,:);
        norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
        log_weight(k,:) = log(norm_weight(k,:));
    end
%    save(strcat('samples',num2str(k),'.mat'),'x_estimate');
end

% Plot the evolution of particles and corresponding marginal distributions
figure; 
% Plot the particles {X_{1:k}} at each time step k
subplot(2,1,2);
for i=1:num_particle
    plot(1:T,xx_estimate(1:T,i));
    hold on;
end
axis([T-50 T -3 2]);
grid on;title('Prob4(f)');
xlabel('time index'); ylabel('state');

% Plot the marginal distribution of p(x_k|y_{1:k}) from the samples
% {X_k^(i)}. The variable x_current records the samples {X_k} at each time
% step and can be used to give the empirical distribution of p(x_k|y_{1:k})

subplot(2,1,1);
XX = -3:0.1:2;
for i=1:T
    ff = ksdensity(x_current(i,:),XX,'width',0.5);
    plot(ff+(i-1),XX);
    hold on;
    sample_den(i) = interp1(XX,ff,X(i));
    plot(sample_den(i)+(i-1),X(i),'ro');
    hold on;
end
axis([T-50 T -3 2]);title('Prob4(f)');
xlabel('time index'); ylabel('state');

%% Prob_4(g)

xx_estimate=zeros(T, num_particle);   % samples of the Markov states
weight=ones(size(xx_estimate));               % importance weights

% Initialization (time step = 1)
x0 = randn(1,num_particle);
sigma = 1/sqrt(1/B^2+C^2/D^2)*ones(1,num_particle);
mu = (sigma.^2).*(A*x0/B^2+C*Y(1)/D^2);
xx_estimate(1,:) = normrnd(mu,sigma,1,num_particle); 
x_origin = xx_estimate;
log_weight(1,:) = log(normpdf(Y(1),C*xx_estimate(1,:),D)) ...
    -log(normpdf(xx_estimate(1,:),mu,sigma)) ...
    +log(normpdf(xx_estimate(1,:),A*x0,B));
lkhmax = max(log_weight(1,:));
weight(1,:) = exp(log_weight(1,:)-lkhmax);
norm_weight(1,:) = weight(1,:)/sum(weight(1,:)); % normalized weight
ess(1) = 1/sum(norm_weight(1,:).^2);             % effective sample size
x_current = xx_estimate;
% save('samples1.mat','x_estimate');

% SMC estimation
for k = 2:T
    sigma = 1/sqrt(1/B^2+C^2/D^2)*ones(1,num_particle);
    mu = (sigma.^2).*(A*xx_estimate(k-1,:)/B^2+C*Y(k)/D^2);
    xx_estimate(k,:) = normrnd(mu,sigma,1,num_particle);
    x_current(k,:) = xx_estimate(k,:);
    log_weight(k,:) = log_weight(k-1,:)+ ...
        log(normpdf(Y(k),C*xx_estimate(k,:),D))- ...
        log(normpdf(xx_estimate(k,:),mu,sigma)) ...
        +log(normpdf(xx_estimate(k,:),A*xx_estimate(k-1,:),B)); 
    lkhmax = max(log_weight(k,:));
    weight(k,:) = exp(log_weight(k,:)-lkhmax);
    norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
    ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
%     cv(k) = sqrt(sum((num_particle*norm_weight(k,:)-1).^2)/num_particle); % coefficient of variance
    
    % Resampling procedure
    x_origin(k,:) = xx_estimate(k,:);
    % make resampling at each time step
    if ess < 0.5*num_particle,
        ind = resampleSystematic(norm_weight(k,:));
        xx_estimate = xx_estimate(:,ind);         
        x_current(k,:) = xx_estimate(k,:);
        norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
        log_weight(k,:) = log(norm_weight(k,:));
    end
%    save(strcat('samples',num2str(k),'.mat'),'x_estimate');
end

% Plot the evolution of particles and corresponding marginal distributions
figure; 
% Plot the particles {X_{1:k}} at each time step k
subplot(2,1,2);
for i=1:num_particle
    plot(1:T,xx_estimate(1:T,i));
    hold on;
end
axis([T-50 T -3 2]);
grid on;title('Prob4(g)');
xlabel('time index'); ylabel('state');

% Plot the marginal distribution of p(x_k|y_{1:k}) from the samples
% {X_k^(i)}. The variable x_current records the samples {X_k} at each time
% step and can be used to give the empirical distribution of p(x_k|y_{1:k})

subplot(2,1,1);
XX = -3:0.1:2;
for i=1:T
    ff = ksdensity(x_current(i,:),XX,'width',0.5);
    plot(ff+(i-1),XX);
    hold on;
    sample_den(i) = interp1(XX,ff,X(i));
    plot(sample_den(i)+(i-1),X(i),'ro');
    hold on;
end
axis([T-50 T -3 2]);title('Prob4(g)');
xlabel('time index'); ylabel('state');