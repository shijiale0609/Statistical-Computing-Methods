% -------------------------------------------------------------------------
% Copyright: MPDC Laboratory, Cornell University, 2010 
% http://mpdc.mae.cornell.edu/ (Written by 
% Meenakshi Sundaram & Prof. N. Zabaras)
%
% Course: MAE7140 Bayesian Scientific Computing
% Remark: This code follows the descriptions and derivations given in the
% lecture notes
%
% This script demonstrates the efficacy of importance sampling to integrate
% a function 
% 0.5*exp(-90*(x-0.5).^2-45*(y+.1).^4)+exp(-45*(x+0.4).^2-60*(y-0.5).^2);
% over the domain [-1 1]x[-1 1]
% -------------------------------------------------------------------------
%%
clear;close all;clc;

%% function to integrate
f=@(x,y) 0.5*exp(-90*(x-0.5).^2-45*(y+.1).^4)+exp(-45*(x+0.4).^2-60*(y-0.5).^2);

%% Proposal Distribution---Mixture of Gaussians
mu1=[0.5,-0.1];Sigma1=[1/180 0; 0 1/20];
mu2=[-0.4,0.5];Sigma2=[1/90 0;0 1/120];
p1=0.46;
p2=1-p1;
g=@(x,y) reshape(p1*mvnpdf([x(:),y(:)],mu1,Sigma1)+p2*mvnpdf([x(:),y(:)],mu2,Sigma2),size(x));

%% Generate Grid
x=linspace(-1,1,101);y=linspace(-1,1,101);
[X,Y]=meshgrid(x,y);

%% Figure Initialization
h=figure(1);clf;set(gca,'fontsize',16);hold on
set(h,'Position',get(0,'Screensize'));
colormap([131 	111 	255;...
          125 	185 	15]/255);
%% Plot proposal distribution
Z=g(X,Y);
figure(1);
h1=surf(X,Y,Z,zeros(size(Z)));
alpha(h1,0.5);

%% Plot function to be integrated
Z=f(X,Y);
figure(1);
surf(X,Y,Z,ones(size(Z)));

%% Legend
h2=legend('Proposal Distribution','Function to be integrated');
set(h2,'location','best');
view(3);

%% Traditional MC Vs Importance Sampling
numSamples=2000;
numCount=200;
MCInt=zeros(numCount,1);
IMPSampInt=zeros(numCount,1);
% To use parfor one must open the matlabpool.
% type in the command matlabpool open 4
for i = 1:numCount
    
    % Traditional Monte Carlo
    % 4 is the area [-1 1]x[-1 1]
    % So one has to multiply that with the expectation
    MCInt(i)=4*sum(f(2*rand(numSamples,1)-1,2*rand(numSamples,1)-1))/numSamples;
    
    % Importance Sampling
    % Sample from the Mixture of Gaussians
    u=rand(numSamples,1);
    % Samples not in [-1 1]x[-1 1] give rise to zero contribution
    % so eliminate them
    sample1=mvnrnd(mu1',Sigma1,sum((u<p1)));
    sample1((sample1(:,1)<-1 | sample1(:,1)>1),:)=[];
    sample1((sample1(:,2)<-1 | sample1(:,2)>1),:)=[];
    
    sample2=mvnrnd(mu2',Sigma2,sum((u>p1)));
    sample2((sample2(:,1)<-1 | sample2(:,1)>1),:)=[];
    sample2((sample2(:,2)<-1 | sample2(:,2)>1),:)=[];
    
    % Expectation of these samples
    IMPSampInt(i)=sum([f(sample1(:,1),sample1(:,2))./g(sample1(:,1),sample1(:,2));...
        f(sample2(:,1),sample2(:,2))./g(sample2(:,1),sample2(:,2))])/numSamples;
    
end

%% Plot of the obtained result
h=figure(2);clf;set(gca,'fontsize',16);
set(h,'Position',get(0,'Screensize'));
plot(1:numCount,[MCInt IMPSampInt],'o');hold on;
plot(1:numCount,dblquad(f,-1,1,-1,1)*ones(1,numCount),'-r','linewidth',2);
xlabel('Number of Trials with 2000 Samples');
ylabel('Integral of f');
grid on;
legend('Standard Monte Carlo','Importance Sampling','Actual Solution');



