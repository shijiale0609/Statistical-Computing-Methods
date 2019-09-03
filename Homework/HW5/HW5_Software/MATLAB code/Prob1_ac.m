%% HW5_Prob1_(a)
clear; close all;
yy = load('logreturns2012to2014.txt'); % Load data: observation (yy), Markov state(xx)
                        % The observation data are simulated from the exact
                        % SV model. First,we generate a Markov chain xx
                        % from the state equation. Then the observation
                        % data yy are obtained from the observation
                        % equation.

% Parameters for the Stochastic Volatility Model
v=0;
fai=0.98;
sigma=0.16;
sigma1 = sigma;
beta = 0.7;
time_step=length(yy);                   % number of time steps
num_particle=500;                       % number of particles 


x_estimate=zeros(time_step, num_particle);      % samples of x_t
weight=ones(size(x_estimate));                  % importance weight


% Initialization (time step = 1)
x_estimate(1,:)=sigma1*randn(1,num_particle);
x_origin=x_estimate;
log_weight(1,:)=-0.5*(yy(1).^2./((beta^2)*exp(x_estimate(1,:))))...
    -0.5*log(2*pi*exp((beta^2)*x_estimate(1,:)))-0.5*x_estimate(1,:).^2/sigma1^2;        % logarithm weight
lkhmax=max(log_weight(1,:));
weight(1,:)=exp(log_weight(1,:)-lkhmax);        
norm_weight(1,:)=weight(1,:)/sum(weight(1,:));  % normalized weight
ess(1)=1/sum(norm_weight(1,:).^2);              % effective sample size
x_current=x_estimate;

% SMC estimation over all time steps
for k=2:time_step
    x_estimate(k,:) = normrnd(v+fai*x_estimate(k-1,:),sigma);
    x_current(k,:) = x_estimate(k,:);
    log_weight(k,:) = log_weight(k-1,:)-0.5*(yy(k).^2./((beta^2)*exp(x_estimate(k,:))))...
        -0.5*log( 2*pi*exp( (beta^2)*x_estimate(k,:) ) );
    lkhmax = max(log_weight(k,:));
    weight(k,:) = exp(log_weight(k,:)-lkhmax);
    norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
    ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
    
    % Resampling procedure
    x_origin(k,:) = x_estimate(k,:);
    
    ind = resampleMultinomial(norm_weight(k,:));
    x_estimate = x_estimate(:,ind);          
    norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
    log_weight(k,:) = log(norm_weight(k,:));
    x_current(k,:) = x_estimate(k,:);

end

% calculate the posterior mean of the particles
x_sample = zeros(time_step,1);
for i = 1:num_particle
    x_sample = x_sample + x_estimate(:,i) * norm_weight(time_step,i);
end

figure;
plot(1:time_step, x_sample,'r','linestyle','--');
xlabel('time step'); ylabel('Markov State vs observation');
hold on;
plot(1:time_step, yy, 'b')
legend('x estimates', 'observation');
axis([0 time_step -10 10]);grid on;title('Prob1(a)');

figure; 
% Plot the particles {X_{1:k}} at each time step k
for i=1:num_particle
    plot(1:time_step,x_estimate(1:time_step,i));
    hold on;
end
axis([1 time_step -2 4]);
grid on;title('Prob1(c) Multinomial Resampling');
xlabel('time index'); ylabel('state');

%% Prob_1_c Systematic_Resampling
x_estimate=zeros(time_step, num_particle);      % samples of x_t
weight=ones(size(x_estimate));                  % importance weight


% Initialization (time step = 1)
x_estimate(1,:)=sigma1*randn(1,num_particle);
x_origin=x_estimate;
log_weight(1,:)=-0.5*(yy(1).^2./((beta^2)*exp(x_estimate(1,:))))...
    -0.5*log(2*pi*exp((beta^2)*x_estimate(1,:)))-0.5*x_estimate(1,:).^2/sigma1^2;        % logarithm weight
lkhmax=max(log_weight(1,:));
weight(1,:)=exp(log_weight(1,:)-lkhmax);        
norm_weight(1,:)=weight(1,:)/sum(weight(1,:));  % normalized weight
ess(1)=1/sum(norm_weight(1,:).^2);              % effective sample size
x_current=x_estimate;

% SMC estimation over all time steps
for k=2:time_step
    x_estimate(k,:) = normrnd(v+fai*x_estimate(k-1,:),sigma);
    x_current(k,:) = x_estimate(k,:);
    log_weight(k,:) = log_weight(k-1,:)-0.5*(yy(k).^2./((beta^2)*exp(x_estimate(k,:))))...
        -0.5*log( 2*pi*exp( (beta^2)*x_estimate(k,:) ) );
    lkhmax = max(log_weight(k,:));
    weight(k,:) = exp(log_weight(k,:)-lkhmax);
    norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
    ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
    
    % Resampling procedure
    x_origin(k,:) = x_estimate(k,:);
    ind = resampleSystematic(norm_weight(k,:));
    x_estimate = x_estimate(:,ind);          
    norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
    log_weight(k,:) = log(norm_weight(k,:));
    x_current(k,:) = x_estimate(k,:);
end


figure; 
% Plot the particles {X_{1:k}} at each time step k
for i=1:num_particle
    plot(1:time_step,x_estimate(1:time_step,i));
    hold on;
end
axis([1 time_step -2 4]);
grid on;title('Prob1(c) Systematic Resampling');
xlabel('time index'); ylabel('state');

%% Part_1(c) ESS Triggered Resampling (when ESS < 0.5*N)

x_estimate=zeros(time_step, num_particle);      % samples of x_t
weight=ones(size(x_estimate));                  % importance weight


% Initialization (time step = 1)
x_estimate(1,:)=sigma1*randn(1,num_particle);
x_origin=x_estimate;
log_weight(1,:)=-0.5*(yy(1).^2./((beta^2)*exp(x_estimate(1,:))))...
    -0.5*log(2*pi*exp((beta^2)*x_estimate(1,:)))-0.5*x_estimate(1,:).^2/sigma1^2;        % logarithm weight
lkhmax=max(log_weight(1,:));
weight(1,:)=exp(log_weight(1,:)-lkhmax);        
norm_weight(1,:)=weight(1,:)/sum(weight(1,:));  % normalized weight
ess(1)=1/sum(norm_weight(1,:).^2);              % effective sample size
x_current=x_estimate;

% SMC estimation over all time steps
for k=2:time_step
    x_estimate(k,:) = normrnd(v+fai*x_estimate(k-1,:),sigma);
    x_current(k,:) = x_estimate(k,:);
    log_weight(k,:) = log_weight(k-1,:)-0.5*(yy(k).^2./((beta^2)*exp(x_estimate(k,:))))...
        -0.5*log( 2*pi*exp( (beta^2)*x_estimate(k,:) ) );
    lkhmax = max(log_weight(k,:));
    weight(k,:) = exp(log_weight(k,:)-lkhmax);
    norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
    ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
    
    % Resampling procedure
    x_origin(k,:) = x_estimate(k,:);
    
    % control the implementation of resampling
    % if set ess(k)>0, then we make resampling at each time step
    % if set ess(k)<0, then we don't make any resampling 
    % In this code, we perform resampling when ESS < 0.3*num_particle
    resample_criterion = num_particle*0.3;
    if ess < resample_criterion   
        ind = resampleMultinomial(norm_weight(k,:));
        x_estimate = x_estimate(:,ind);          
        norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
        log_weight(k,:) = log(norm_weight(k,:));
        x_current(k,:) = x_estimate(k,:);
    end
end

figure; 
% Plot the particles {X_{1:k}} at each time step k
for i=1:num_particle
    plot(1:time_step,x_estimate(1:time_step,i));
    hold on;
end
axis([1 time_step -2 4]);
grid on;title('Prob1(c) Multinomial Resampling (ESS Triggered, ESS < 0.5N)');
xlabel('time index'); ylabel('state');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_estimate=zeros(time_step, num_particle);      % samples of x_t
weight=ones(size(x_estimate));                  % importance weight


% Initialization (time step = 1)
x_estimate(1,:)=sigma1*randn(1,num_particle);
x_origin=x_estimate;
log_weight(1,:)=-0.5*(yy(1).^2./((beta^2)*exp(x_estimate(1,:))))...
    -0.5*log(2*pi*exp((beta^2)*x_estimate(1,:)))-0.5*x_estimate(1,:).^2/sigma1^2;        % logarithm weight
lkhmax=max(log_weight(1,:));
weight(1,:)=exp(log_weight(1,:)-lkhmax);        
norm_weight(1,:)=weight(1,:)/sum(weight(1,:));  % normalized weight
ess(1)=1/sum(norm_weight(1,:).^2);              % effective sample size
x_current=x_estimate;

% SMC estimation over all time steps
for k=2:time_step
    x_estimate(k,:) = normrnd(v+fai*x_estimate(k-1,:),sigma);
    x_current(k,:) = x_estimate(k,:);
    log_weight(k,:) = log_weight(k-1,:)-0.5*(yy(k).^2./((beta^2)*exp(x_estimate(k,:))))...
        -0.5*log( 2*pi*exp( (beta^2)*x_estimate(k,:) ) );
    lkhmax = max(log_weight(k,:));
    weight(k,:) = exp(log_weight(k,:)-lkhmax);
    norm_weight(k,:) = weight(k,:)/sum(weight(k,:));
    ess = 1/sum(norm_weight(k,:).^2);                                  % effective sample size
    
    % Resampling procedure
    x_origin(k,:) = x_estimate(k,:);
    
    
    resample_criterion = num_particle*0.3;
    if ess < resample_criterion   
        ind = resampleSystematic(norm_weight(k,:));
        x_estimate = x_estimate(:,ind);          
        norm_weight(k,:) = 1/num_particle*ones(1,num_particle);
        log_weight(k,:) = log(norm_weight(k,:));
        x_current(k,:) = x_estimate(k,:);
    end
end


figure; 
% Plot the particles {X_{1:k}} at each time step k
for i=1:num_particle
    plot(1:time_step,x_estimate(1:time_step,i));
    hold on;
end
axis([1 time_step -2 4]);
grid on;title('Prob1(c) Systematic Resampling (ESS Triggered, ESS < 0.5N)');
xlabel('time index'); ylabel('state');

%% 

