%% HW5_Prob1_(b) Resampling vs No Resampling
clear all;

% Parameters for the Stochastic Volatility Model
fai=0.98;
sigma=0.16;
                   % number of time steps
num_particle=500;                       % number of particles 
yy = load('logreturns2012to2014.txt'); % Load data: observation (yy), Markov state(xx)
time_step=length(yy);
beta = 0.25:0.25:2;
likelihood = zeros(length(beta), 10);
for n = 1:length(beta)
    for l = 1:10
        x_estimate=zeros(time_step, num_particle);      % samples of x_t
        weight=ones(size(x_estimate));                  % importance weight


        % Initialization (time step = 1)
        x_estimate(1,:)=sigma*randn(1,num_particle);
        x_origin=x_estimate;
        log_weight(1,:)=-0.5*(yy(1).^2./((beta(n)^2)*exp(x_estimate(1,:))))...
            -0.5*log(2*pi*exp((beta(n)^2)*x_estimate(1,:)));        % logarithm weight
        lkhmax=max(log_weight(1,:));
        weight(1,:)=exp(log_weight(1,:)-lkhmax);        
        norm_weight(1,:)=weight(1,:)/sum(weight(1,:));  % normalized weight
        ess(1)=1/sum(norm_weight(1,:).^2);              % effective sample size
        x_current=x_estimate;

        % SMC estimation over all time steps
        for k=2:time_step
            x_estimate(k,:) = normrnd(fai*x_estimate(k-1,:),sigma);
            x_current(k,:) = x_estimate(k,:);
            log_weight(k,:) = log_weight(k-1,:)-0.5*(yy(k).^2./((beta(n)^2)*exp(x_estimate(k,:))))...
                -0.5*log( 2*pi*exp( (beta(n)^2)*x_estimate(k,:) ) );
            lkhmax = max(log_weight(k,:));
            weight(k,:) = exp(log_weight(k,:)-lkhmax);
            norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
%             ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
% 
%             % Resampling procedure
            x_origin(k,:) = x_estimate(k,:);

            ind = resampleMultinomial(norm_weight(k,:));
            x_estimate = x_estimate(:,ind);          
            norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
            log_weight(k,:) = log(norm_weight(k,:));
            x_current(k,:) = x_estimate(k,:);

        end
        likelihood(n,l) = sum(log(sum(weight,2))-log(num_particle));
    end
end
boxplot(likelihood', beta); grid on;
title('Prob1(b): Likelihood box plot vs \beta (Resampling)');
xlabel('\beta');ylabel('likelihood');


clear all;

% Parameters for the Stochastic Volatility Model
fai=0.98;
sigma=0.16;
                   % number of time steps
num_particle=500;                       % number of particles 
yy = load('logreturns2012to2014.txt'); % Load data: observation (yy), Markov state(xx)
time_step=length(yy);
beta = 0.25:0.25:2;
likelihood = zeros(length(beta), 10);
for n = 1:length(beta)
    for l = 1:10
        x_estimate=zeros(time_step, num_particle);      % samples of x_t
        weight=ones(size(x_estimate));                  % importance weight


        % Initialization (time step = 1)
        x_estimate(1,:)=sigma*randn(1,num_particle);
        x_origin=x_estimate;
        log_weight(1,:)=-0.5*(yy(1).^2./((beta(n)^2)*exp(x_estimate(1,:))))...
            -0.5*log(2*pi*exp((beta(n)^2)*x_estimate(1,:)));        % logarithm weight
        lkhmax=max(log_weight(1,:));
        weight(1,:)=exp(log_weight(1,:)-lkhmax);        
        norm_weight(1,:)=weight(1,:)/sum(weight(1,:));  % normalized weight
        ess(1)=1/sum(norm_weight(1,:).^2);              % effective sample size
        x_current=x_estimate;

        % SMC estimation over all time steps
        for k=2:time_step
            x_estimate(k,:) = normrnd(fai*x_estimate(k-1,:),sigma);
            x_current(k,:) = x_estimate(k,:);
            log_weight(k,:) = log_weight(k-1,:)-0.5*(yy(k).^2./((beta(n)^2)*exp(x_estimate(k,:))))...
                -0.5*log( 2*pi*exp( (beta(n)^2)*x_estimate(k,:) ) );
            lkhmax = max(log_weight(k,:));
            weight(k,:) = exp(log_weight(k,:)-lkhmax);
            norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
%             ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
% 
%             % Resampling procedure
            x_origin(k,:) = x_estimate(k,:);
% 
%             ind = polyrnd(norm_weight(k,:),num_particle);
%             x_estimate = x_estimate(:,ind);          
%             norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
%             log_weight(k,:) = log(norm_weight(k,:));
%             x_current(k,:) = x_estimate(k,:);

        end
        likelihood(n,l) = sum(log(sum(weight,2))-log(num_particle));
    end
end
figure;
boxplot(likelihood', beta); grid on;
title('Prob1(b): Likelihood box plot vs \beta (No Resampling)');
xlabel('\beta');ylabel('likelihood');