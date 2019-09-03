%% Prob_4_abcd
clear;
clc;close all;
%% Part (a)
timestep = 2000;
X = zeros(1,timestep);
Y = zeros(1,timestep);
p0 = 1;
X0 = sqrt(p0)*randn;
Q = 0.1; R = 0.1;
A = 0.7; C = 0.5;
X(1) = A*X0+sqrt(Q)*randn;
Y(1) = C*X(1)+sqrt(R)*randn;

for i = 2:timestep
    X(i) = A*X(i-1)+sqrt(Q)*randn;
    Y(i) = C*X(i)+sqrt(R)*randn;
end

%% Part (b)
X_tt = zeros(1,timestep);
P_tt = zeros(1,timestep);
P_tt(1) = p0;
X_tt(1) = 0;

for i = 2:timestep
    Ptt_1 = A*P_tt(i-1)*A+Q;
    Kt = Ptt_1*C/(C*Ptt_1*C+R);
    X_tt(i) = A*X_tt(i-1)+Kt*(Y(i)-C*A*X_tt(i-1));
    P_tt(i) = Ptt_1-Kt*C*Ptt_1;
end
figure;
plot(timestep-100:timestep, X(timestep-100:timestep),'-.', (timestep-100):timestep, X_tt(timestep-100:timestep), 'linewidth',2);grid on;
legend('True State', 'Kalman Filter estimate');
xlabel('Time, t');ylabel('x_t');
title('comparison of Kalmann estimate with True state (N = 500)');

%% Part (c)
timestep = 2000;
Number = [100 500 1000];
Mean_Boot = zeros(length(Number),timestep);
variance = zeros(length(Number),timestep);
for j = 1:length(Number)
    num_particle = Number(j);
    x = ones(timestep,num_particle);
    w = x;
    x(1,:) = sqrt(p0)*randn(1,num_particle);
    for i = 2:timestep+1
        x(i,:) = A*x(i-1,:)+sqrt(Q)*randn(1,num_particle);
        weights = normpdf(Y(i-1),C*x(i,:),sqrt(R)*ones(1,num_particle));
        weights = weights/sum(weights);
        Mean_Boot(j,i-1) = sum(weights.*x(i,:));
        variance(j,i-1) = sum(weights.*(x(i,:).^2))-Mean_Boot(j,i-1).^2;
        rs = randsample(1:num_particle, num_particle, true, weights);
        temp = x(i,rs);
        x(i,:) = temp; 
    end
end

disp('Bootstrap vs. Kalman Average absolute difference in mean estimates:\n');

for j = 1:length(Number)
    fprintf('For N = %d: %2.4f\n', Number(j), mean(abs(X_tt-Mean_Boot(j,:))));
end

disp('Average absolute difference in variance estimates:\n');
for j = 1:length(Number)
    fprintf('For N = %d: %2.4f\n', Number(j), mean(abs(P_tt-variance(j,:))));
end

%% Part (d)
Mean_FA = zeros(length(Number),timestep);
Variance_FA = zeros(length(Number),timestep);
sig_n = 1./(sqrt(Q)^-2+(sqrt(R)/C)^-2);
for j = 1:length(Number)
    num_particle = Number(j);
    x = ones(timestep,num_particle);
    w = x;
    x(1,:) = sqrt(p0)*randn(1,num_particle); % Generate N particles
    for i = 2:timestep+1
        mu_i = sig_n*(A*x(i-1,:)/Q+C*Y(i-1)/R);
        x(i,:) = mu_i+sqrt(sig_n)*randn(1,num_particle); % Propagation
        temp1 = normpdf(Y(i-1),C*x(i,:),sqrt(R)*ones(1,num_particle));
        temp2 = normpdf(x(i,:), A*x(i-1,:), sqrt(Q)*ones(1,num_particle));
        temp3 = normpdf(x(i,:), mu_i, sqrt(sig_n)*ones(1,num_particle));
        weights = temp1.*temp2./temp3;
        weights = weights/sum(weights);
        Mean_FA(j,i-1) = sum(weights.*x(i,:));
        Variance_FA(j,i-1) = sum(weights.*(x(i,:).^2))-Mean_FA(j,i-1).^2;
        rs = randsample(1:num_particle, num_particle, true, weights); % Resampling
        temp = x(i,rs);
        x(i,:) = temp; 
    end
end


figure;
plot(timestep-100:timestep, X(timestep-100:timestep),'-.', (timestep-100):timestep, X_tt(timestep-100:timestep), timestep-100:timestep,...
    Mean_Boot(2, timestep-100:timestep),timestep-100:timestep, Mean_FA(2, timestep-100:timestep), 'linewidth',2);grid on;
legend('True State', 'Kalman Filter estimate', 'Bootstrap estimate', 'Fully Adapted');
xlabel('Time, t');ylabel('x_t');
title('comparison of Kalmann, Bootstrap and Fully Adapted with True state (N = 500)');


%%
fprintf('Kalman Filter vs. Fully Adapted Particle Filter\n');
fprintf('Average absolute difference in mean estimates\n');
for j = 1:length(Number)
    fprintf('For N = %d: %2.4f\n', Number(j), mean(abs(X_tt-Mean_FA(j,:))));
end

fprintf('Average absolute difference in variance estimates:\n');
for j = 1:length(Number)
    fprintf('For N = %d: %2.4f\n', Number(j), mean(abs(P_tt-Variance_FA(j,:))));
end


fprintf('Bootstrap Particle Filter vs. Fully Adapted Particle Filter\n');
fprintf('Average absolute difference in mean estimates\n');
for j = 1:length(Number)
    fprintf('For N = %d: %2.4f\n', Number(j), mean(abs(Mean_Boot(j,:)-Mean_FA(j,:))));
end

fprintf('Average absolute difference in variance estimates:\n');
for j = 1:length(Number)
    fprintf('For N = %d: %2.4f\n', Number(j), mean(abs(variance(j,:)-Variance_FA(j,:))));
end