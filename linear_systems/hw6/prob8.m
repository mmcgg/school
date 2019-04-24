close all; clear all; clc;
A = [-.038, 18.984, 0, -32.174;
     -.001, -.632, 1, 0;
     0, -.759, -.518, 0;
     0, 0, 1, 0];
 
B = [10.1, 0;
     0, -.0086;
     .025, -.011;
     0, 0];
 
C = [1 0 0 0];
D = [];

sys = ss(A,B,C,D);

Q1 = diag([1/250^2 1/deg2rad(.05)^2 1/deg2rad(.05)^2 1/deg2rad(.05)^2]);
R1 = diag([1/.5^2 1/deg2rad(25)^2]);

[K,S,E] = lqr(A,B,Q1,R1*1000);
[wn,zeta1] = damp(E);
[K,S,E] = lqr(A,B,Q1,R1*100);
[wn,zeta2] = damp(E);
[K,S,E] = lqr(A,B,Q1,R1*10);
[wn,zeta3] = damp(E);
[K,S,E] = lqr(A,B,Q1,R1*7);
[wn,zeta4] = damp(E);

% figure(1)
% plot(zeta1)
% hold on
% plot(zeta2)
% plot(zeta3)
% plot(zeta4)

%% Part B
x0 = [20 .01 -.01 .02]';

Q = Q1;
R = R1*100;
K = lqr(A,B,Q,R);
x = x0;
xhist = [];
uhist = [];
dt=.01;
t_final = 10;
for t=0:dt:t_final
    u = -K*x;
    
    xdot = A*x + B*u;
    x = x +xdot*dt;
    
    xhist = [xhist,x];       
    uhist = [uhist,u];           
end
t=0:dt:t_final;
figure(2)
plot(t,xhist)

figure(3)
plot(t,uhist)
    
%% Part C
C = [1 0 0 0;
     0 -1 0 1];
% Duality of LQR and Kalman Filter
[Llqr, Slqr, Elqr] = lqr(A',C',1e-4*B*B',diag([1 1e-5]));

[Lkf, Skf, Ekf] = lqe(A,eye(4),C,1e-4*B*B',diag([1 1e-5]));


%% Parts F and G
Q = Q1;
R = R1*1000;
K = lqr(A,B,Q,R);
L = Lkf;

x = 1.0*[1;1;1;1];
xdes = [.5,0,0,0]';
xhat = [1;1;1;1] + .01*randn(4,1);
xhist = [];
uhist = [];
xhathist = [];
ehist = [];
dt=.001;
t_final = 50;
for t=0:dt:t_final
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
    uhist = [uhist,u];

    yhat = C*xhat;
    xhatdot = A*xhat + B*u - (L*(yhat-y));
    xhat = xhat + xhatdot*dt;

    xhathist = [xhathist, xhat];
    
    ehist = [ehist, x-xhat];    
end
t=0:dt:t_final;
figure(5)
plot(t,xhist,'b')
hold on
plot(t,xhathist,'r')

figure(7)
plot(t,uhist)

figure(6)
plot(t,ehist)

% This approach seems reasonable, but then again there is the whole
% linearization assumption. For my problem, the 