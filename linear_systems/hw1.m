%% Inverted Pendulum feedback linearization
close all; clear all; clc;
l = 1;
m = 1;
b = .1;
g = 9.8;
Kp = 1;
Kd = .9;

Ts = [];
thetas = [];
dthetas = [];
dt = .001;

theta_des = 0;
dtheta = 0;
theta = pi;

for t=0:dt:10

    % Calculate Feedback linearizing control input
    T = m*l^2*(Kp*(theta_des-theta)-Kd*dtheta + b/(m*l^2) - g/l*sin(theta));
    
    ddtheta = g/l*sin(theta) - b/(m*l^2)*dtheta + T/(m*l^2);
    dtheta = dtheta + ddtheta*dt;
    theta = theta + dtheta*dt;
    
    Ts = [Ts; T];
    thetas = [thetas; theta];
    dthetas = [dthetas; dtheta];
end

plot(thetas)
hold on
plot(Ts)
legend('theta','torque')

%% Linearized aircraft dynamics

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

sys = ss(A,B,C,D);
[y,t] = step(sys);

figure()
plot(t,squeeze(y(:,:,1)))
legend('V','angle of attack', 'pitch rate', 'pitch')
title('step in elevators')

figure()
plot(t,squeeze(y(:,:,2)))
legend('V','angle of attack', 'pitch rate', 'pitch')
title('step in thrust')

max(real(eig(A)))

% This system is stable because all eigenvalues of A have negative real
% part