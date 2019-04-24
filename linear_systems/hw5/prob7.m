close all; clear all; clc;
A = [-.038, 18.984, 0, -32.174;
     -.001, -.632, 1, 0;
     0, -.759, -.518, 0;
     0, 0, 1, 0];
 
B = [10.1, 0;
     0, -.0086;
     .025, -.011;
     0, 0];
 
C = eye(4);
D = [];

%% Part A - assuming C = [1,0,0,0] is the system observable

C = [1 0 0 0];

% This is rank 4, so it is observable 
rank(obsv(A,C));

% Detectable just means that unobservable states are stable, but all of the
% states are observable, so it is also detectable

%% Part B - 
Q = 100.0*eye(4);
R = eye(2);
K = lqr(A,B,Q,R);
L = -100.01*[1;0;0;0];
x = 1.0*[1;1;1;1];
xhat = [1;1;1;1] + .1*randn(4,1);
xhist = [];
xhathist = [];
ehist = [];
dt=.01;
for t=0:dt:100
    u = -K*xhat;
    
    y = C*x;
    xdot = A*x + B*u;
    x = x +xdot*dt;
    
    xhist = [xhist,x];

    yhat = C*xhat;    
    xhatdot = A*xhat + B*u + L*(yhat-y);
    xhat = xhat + xhatdot*dt;
    xhathist = [xhathist, xhat];
    
    ehist = [ehist, x-xhat];    
end
t=0:dt:100;
figure(1)
plot(t,xhist,'b')
hold on
plot(t,xhathist,'r')

figure(2)
plot(t,ehist)

%% Parts C, D, and E
poles = [-5+j,-5-j,-3+.14j,-3-.14j];
K = place(A,B,poles);
L = place(A',C',poles*10);

x = 1.0*[1;1;1;1];
xhat = [1;1;1;1] + .01*randn(4,1);
xhist = [];
xhathist = [];
ehist = [];
dt=.001;
for t=0:dt:10
    u = -K*xhat;
    
    y = C*x;
    xdot = A*x + B*u;
    x = x + xdot*dt;
    
    xhist = [xhist,x];

    yhat = C*xhat;
    xhatdot = A*xhat + B*u - (L*(yhat-y))';
    xhat = xhat + xhatdot*dt;

    xhathist = [xhathist, xhat];
    
    ehist = [ehist, x-xhat];
B = eye(2);
C = [-1,1];
D = [2,1];

sys = ss(A,B,C,D);

msys = minreal(sys);

msys.A
msys.B
msys.C
msys.D;    
end
t=0:dt:10;
figure(3)
plot(t,xhist,'b')
hold on
plot(t,xhathist,'r')

figure(4)
plot(t,ehist)

%% Parts F and G
poles = [-5+j,-5-j,-3+.14j,-3-.14j];
K = place(A,B,poles);
L = place(A',C',poles*10);

x = 1.0*[1;1;1;1];
xdes = [.5,0,0,0]';
xhat = [1;1;1;1] + .01*randn(4,1);
xhist = [];
xhathist = [];
ehist = [];
dt=.001;
for t=0:dt:10
    if t>5
        xdes = [.5,0,0,0]';
    else
        xdes = [0,0,0,0]';
    end
    u = K*(xdes-xhat);
    
    y = C*x;
    xdot = A*x + B*u;
    x = x + xdot*dt;
    
    xhist = [xhist,x];

    yhat = C*xhat;
    xhatdot = A*xhat + B*u - (L*(yhat-y))';
    xhat = xhat + xhatdot*dt;

    xhathist = [xhathist, xhat];
    
    ehist = [ehist, x-xhat];    
end
t=0:dt:10;
figure(5)
plot(t,xhist,'b')
hold on
plot(t,xhathist,'r')

figure(6)
plot(t,ehist)

% This approach seems reasonable, but then again there is the whole
% linearization assumption. For my problem, the 