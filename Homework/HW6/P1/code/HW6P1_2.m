% this is demo1.m
% this is demo1.m
clc;
clear all;

load data2train.dat;
%data2train
%rng(2)
%y = genmix(900,mu,covar,pp);
%clear covar mu mu1 mu2 mu3
[bestk,bestpp,bestmu,bestcov,dl,countf] = mixtures4(data2train.',1,25,0,1e-4,1)
