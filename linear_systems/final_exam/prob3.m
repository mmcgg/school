%% Part A
% Check the eigenvalues of A. One is positive => unstable.
A = [-2 4;
     1 1];
 
eig(A)

%% Part B
% Verify that the given state transition matrix matches a
% numerical approximation of the solution to the differential equation. It
% looks like it does.
close all;
x0 = randn(2,1);
[t,y] = ode45(@odefunc,[0,1000],x0);

my_y = zeros(size(y));
for i=1:length(t)
    Phi = [1 0;
           .5*t(i)^2 1];
    my_y(i,:) = Phi*x0;
end

plot(t,y,'b')
hold on
plot(t,my_y,'r--')
title('Numerical Solution vs analytical solution')

figure()
plot(t,my_y - y)
title('Difference between analytical and numerical solutions')

%% Part d
% Check the rank of the observability matrix for this (A,C) pair at several
% values of T. The rank never goes above 1 and the entries with zeros in
% them never change. This means it can't be fully observable.
for t=0:.001:100
    A = [0 0;
         t 0];
    C = [exp(-2*t) 0];
    if(rank(obsv(A,C))>1)
        disp("Rank more than one!")
    end
end

%% ODE function for part b
function xdot = odefunc(t,x)
    A = [0 0;
         t 0];
    xdot = A*x;
end    