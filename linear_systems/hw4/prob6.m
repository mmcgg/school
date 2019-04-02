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

%% Part A - show the system is controllable with only elevator
% This is equal to 4, so the system is controllable with only elevator
rank(ctrb(A,B(:,2)))

%% Part B - Throttle sticking in race

V0 = 250;
alpha0=0;
q0=0;
theta0=0;

x0 = [V0;alpha0;q0;theta0];

V1 = 300;
alpha1=10*pi/180;
q1=2*pi/180;
theta1=5*pi/180;

x1 = [V1;alpha1;q1;theta1];

xd = x1-x0

x_ss = -inv(A)*B(:,2)

% Because xd is not a constant multiple of x_ss (the steady states we can
% reach with constant input), we cannot permanently acheive xd using only a
% constant elevator

%% Part C and D - Time varying control to acheive xd
close all;
sys = ss(A,B,C,D);
dt = .01;
sysd = c2d(sys,dt);
Ad = sysd.A;
Bd = sysd.B;

Q = 1000*diag([.001,1,1,1]);
R = 0.0001*eye(2);
K = dlqr(Ad,Bd,Q,R);

x = x0;
x_hist = [];
u_hist = [];

for t=0:dt:60
    
    u = -K*(x-xd);
    x_hist = [x_hist,x];
    u_hist = [u_hist,u];
    
    x = Ad*x + Bd*u;
end

t = 0:dt:60;

figure(1)
for i=1:4
    subplot(4,1,i)
    plot(t,x_hist(i,:))
    hold on
    plot(t,ones(size(t))*xd(i),'r')
end


figure(2)
plot(t,u_hist)

x-xd

%% Part E - is it possible to maintain this state indefinitely?

% No - there are no inputs that keep xdot=0 given those states

%% Part F

% It is hard to say how feasible this is, given we have gone far from our
% linearization and don't know if the dynamics are actually represnted by
% these linear time invariant dynamics