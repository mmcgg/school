close all; clear all;clc;
zeta = 2;
wn = 1.5;
dt = .001;
A = [-2*zeta*wn -wn^2;
     -1 0];
 
B = [1;0];

x = [2;
     1];

xhat = [0;
        0];

%% Part A
u = 0;
xhist = [];
xhathist = [];
ehist = [];
L = 1.0*eye(2);
for t=0:dt:15
    xdot = A*x + B*u;
    x = x +xdot*dt;
    xhist = [xhist,x];
    
    xhatdot = A*xhat + B*u + L*(x-xhat);
    xhat = xhat + xhatdot*dt;
    xhathist = [xhathist, xhat];
    
    ehist = [ehist, x-xhat];
end
t=0:dt:15;
figure(1)
plot(t,xhist,'b')
hold on
plot(t,xhathist,'r')

figure(3)
plot(t,ehist)

eig(A-L)

%% Part B
x = [2;
     1];
xhat = [0;
        0];
xhist = [];
xhathist = [];
ehist = [];
L = 1.0*eye(2);
for t=0:dt:15
    u = 3 + .5*sin(.75*t);
    xdot = A*x + B*u;
    x = x +xdot*dt;
    xhist = [xhist,x];
    
    xhatdot = A*xhat + B*u + L*(x-xhat);
    xhat = xhat + xhatdot*dt;
    xhathist = [xhathist, xhat];
    
    ehist = [ehist, x-xhat];    
end
t=0:dt:15;
figure(2)
plot(t,xhist,'b')
hold on
plot(t,xhathist,'r')

figure(4)
plot(t,ehist)

%% Part C

% The input doesn't seem to affect the estimation much. Probably because my
% model for B is perfect and I am able to measure state perfectly as well.