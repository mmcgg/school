close all; clear all; clc;
%% Define the dynamic system from the equation m*vdot + b*v = F
b = 20;
m = 100;

% v = xdot, vdot = xddot
% My state vector is [xdot; x]

A = [-b/m, 0;
     1, 0];

B = [1/m; 0];

C = [0 1];
 
D = 0;

proc_cov = [1.1, 0.0;
            0, .1];
        
meas_cov = [.1];

% Discretize the system dynamics
sim_dt = .05;

% discretizing = expm([A,B;zeros(1,3)]*sim_dt);
% A_d = discretizing(1:2,1:2); 
% B_d = discretizing(1:2,3);

% Other way of discretizing dynamics
sys = ss(A,B,C,D);
sys_d = c2d(sys,sim_dt);
A_d = sys_d.A;
B_d = sys_d.B;


%% This is a simulation
x = [0;0];
t = 0;

x_history = [];
input_history = [];
x_est_history = [];
kalman_gain_history = [];
sigma_history = [];

x_est = [0;0];
sigma = 1e-6*eye(2);

while t<=50

    if(t<5)
        F = 50;
    elseif(t>25 && t<30)
        F = -50;
    else
        F = 0;
    end
    x = A_d*x + B_d*F + sqrt(proc_cov)*randn(2,1);

    measurement = C*x + sqrt(meas_cov)*randn(1)';
    [x_est, sigma,sigma_bar, kalman_gain] = kalman_filter(x_est,sigma,F,measurement, A_d, B_d, C, D, proc_cov, meas_cov);
    
    
    x_history = [x_history, x];
    input_history = [input_history, F];
    x_est_history = [x_est_history, x_est];
    kalman_gain_history = [kalman_gain_history, kalman_gain];
    sigma_history = [sigma_history, [sigma(1);sigma(4)]];
    
    t = t + sim_dt;
end


%% This is plotting
% t = 0:sim_dt:50;
% 
% figure()
% plot(t,x_history')
% hold on
% plot(t,x_est_history')
% legend('xdot','x','xdot est','x est')
% title('States and State Estimates')
% 
% figure()
% plot(t,kalman_gain_history')
% title('Kalman Gains v Time')
% 
% figure()
% plot(t,sigma_history')
% title('Covariance v Time')
% 
% error = x_history-x_est_history;
% 
% figure()
% plot(t,error)
% title('Estimation Error v Time')

%% Load the solution data and compare to Dr. Mclain's solution

load('hw1_soln_data.mat')

x_est = mu0;
sigma = Sig0;
proc_cov = R;
meas_cov = Q;


x_history = [];
input_history = [];
x_est_history = [];
kalman_gain_history = [];
sigma_history = [.1];

for i=1:length(t)
    F = u(i);
    
    x = [vtr(i); xtr(i)];
    
    x_history = [x_history, x];
    input_history = [input_history, F];
    x_est_history = [x_est_history, x_est];
    

    measurement = z(i);
    [x_est, sigma,sigma_bar, kalman_gain] = kalman_filter(x_est,sigma,F,measurement, A_d, B_d, C, D, proc_cov, meas_cov);
    
    sigma_history = [sigma_history, sigma_bar(4), sigma(4)];
    
    kalman_gain_history = [kalman_gain_history, kalman_gain];
    
    
    t = t + sim_dt;
end


%% This is plotting
t = 0:sim_dt:50;

figure()
plot(t,x_est_history')
hold on
plot(t,x_history')
legend('xdot est','x est','xdot','x')
title('Solution States and State Estimates')

figure()
plot(t,kalman_gain_history')
title('Solution Kalman Gains v Time')

figure()
plot(sigma_history(1:200))
title('Solution Covariance v Time')

error = x_history-x_est_history;

figure()
subplot(2,1,1)
plot(t,error(1,:))
subplot(2,1,2)
plot(t,error(2,:))
title('Solution Estimation Error v Time')