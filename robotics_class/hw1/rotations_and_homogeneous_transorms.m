close all; clear all; clc;
%problem 1b)
R = rot2(pi/2); %make a 2D rotation matrix
trplot2(R); %plot it

vec = [1; 0]; %make a vector
transformed_vec = R*vec %transform the vector

inv(R)*R % check the inverse times the original
R*inv(R) % reverse order
det(R)   % find the determinant
R'       % find the inverse (which is same as transpose)

%% problem 1c) - same but for 3D
R = rotx(pi/2); %make a 2D rotation matrix
tranimate(R); %plot it

vec = [1; 0; 0]; %make a vector
transformed_vec = R*vec %transform the vector

inv(R)*R % check the inverse times the original
R*inv(R) % reverse order
det(R)   % find the determinant
R'       % find the inverse (which is same as transpose)


%% problem 1d) 
T = rt2tr(rotx(pi/2), [1, 1, 1]);  %create T
tranimate(T)          %animate T
vec = [1, 0, 0, 1]';  %make a vector to transform
T*vec                 %transform a point

inv(T)*T % check the inverse times the original
T*inv(T) % reverse order