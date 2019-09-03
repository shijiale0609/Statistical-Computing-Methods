% this is demo1.m
% this is demo1.m
clc;
clear all;

load data1train.dat;
%data1train

%y = genmix(900,mu,covar,pp);
%clear covar mu mu1 mu2 mu3
[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(data1train.',1,25,0,1e-4,1)
