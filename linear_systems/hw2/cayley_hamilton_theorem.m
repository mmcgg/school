clc; clear all; close all;
%% Calculating A^t using a basis set of matrices
A1 = [1 1 0;
      0 1 0;
      0 0 1];

for t=1:100
    a0 = 1-t;
    a1 = t;
    
    error = A1^t - (a1*A1 + a0*eye(3))
end

%% Calculating e^(At) using a basis set

for t=1:100
    a0 = 1-t;
    a1 = t;
    
    error = expm(A1*t) - exp(t)*(a1*A1 + a0*eye(3))
end

%% Calculating A^t using a basis set of matrices
A2 = [1 1 0;
      0 0 1;
      0 0 1];

for t=1:100
    a1 = 2-t;
    a2 = t-1;
    
    error = A2^t - (a2*A2^2 + a1*A2)
end
%% Doing the rest with Matlab
syms t
eA2t = expm(A2*t)

A3 = [2 0 0 0;
      2 2 0 0;
      0 0 3 3;
      0 0 0 3];
  
A3t = A3^t

eA3t = expm(A3*t)
