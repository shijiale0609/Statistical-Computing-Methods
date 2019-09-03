%---------------------------------------------------
% Statistical Computing for Scientists and Engineers
% Homework 6
% Fall 2018
% University of Notre Dame
%---------------------------------------------------

clear all;
close all;
clc;

% This function gives height of the submerged train from the bottom of the
% ocean.
ocean_bathymetry = @(x) (x>=10).*((1-(x-10)/30).*sin(x-10)+((x-10)/30).*...
    sin(1.5*(x-10))+0.2.*(x-10).*(x<=20)+2*(x>20))+(x<=-10).*((1-(-x-10)/30).*...
    sin(-x-10)+((-x-10)/30).*sin(1.5*(-x-10))+0.2.*(-x-10).*(x>=-20)+2*(x<-20));


ocean_bathymetry(-25)