close all; clear all; clc;
%% Implementing the velocity motion model
alpha = 5*[.1, .01, .01, .1];
sigma_range = .1;
sigma_bearing = .05;

dt = .1;
x = -5;
y = -3;
theta = 90*pi/180;
landmarks = [-7 8;
              6 4;
              6 -4];
             
%landmarks = randn(100,2)*10;
         
num_landmarks = size(landmarks);
         
range = zeros([1,1]);
bearing = zeros([1,1]);

t = 0:dt:20;

v_hist = [];
w_hist = [];
state_hist = [];
state_est_hist = [];
sigma_hist = [];
range_hist = [];
bearing_hist = [];
K_hist = [];


sigma = .01*eye(3);
mu = [x;y;theta];

figure(1)
landmarks_iterator = 1;
for i=1:length(t)
    
    % Calculate the commanded linear and angular velocities
    v = 1 + .5*cos(2*pi*(.2*t(i)));
    w = -.2 + 2.0*cos(2*pi*(.6*t(i)));
    
    % These are the actual linear and angular velocities
    vhat = v + alpha(1)*v^2*randn() + alpha(2)*w^2*randn();
    what = w + alpha(3)*v^2*randn() + alpha(4)*w^2*randn();
    
    
    v_hist = [v_hist;v];
    w_hist = [w_hist;w];
    
    % Integrate the velocities to get position and heading (this allows
    % straight lines, but assumes move straight, then turn)
%     x = x + v*cos(theta)*dt;
%     y = y + v*sin(theta)*dt;
%     theta = theta + w*dt;
    
    % Integrate the velocities to get position and heading
    % Radius motion model (from pg 126 in Probabilistic Robotics book)
    x = x - vhat/what*sin(theta) + vhat/what*sin(theta+what*dt);
    y = y + vhat/what*cos(theta) - vhat/what*cos(theta+what*dt);    
    theta = theta + what*dt;    
    state_hist = [state_hist; x y theta];
    
    % Get measurements from landmark (range and bearing)
    range_reading = landmarks(landmarks_iterator,:) - [x,y] + sigma_range*randn();
    range = norm(range_reading);
    bearing = atan2(range_reading(2),range_reading(1)) - theta + sigma_bearing*randn();
    
    u = [v;w];
    z = [range';bearing'];
    m = landmarks(landmarks_iterator,:);
    
    range_hist = [range_hist; range'];
    bearing_hist = [bearing_hist; bearing'];
    
    % Get an estimate of state using an EKF
    [mu,sigma,K] = UKF_localization(mu, sigma, u, z, m, dt, alpha, sigma_range, sigma_bearing);
    
    state_est_hist = [state_est_hist; mu'];
    sigma_hist = [sigma_hist; sigma(1,1) sigma(2,2) sigma(3,3)];
    K_hist = [K_hist; K(1,1) K(1,2) K(2,1) K(2,2) K(3,1) K(3,2)];
    
    landmarks_iterator = landmarks_iterator + 1;
    if(landmarks_iterator > num_landmarks)
        landmarks_iterator = 1;
    end
    
end

% Draw the robot
drawRobot(state_hist(:,1),state_hist(:,2),state_hist(:,3),landmarks,range_hist,bearing_hist,t)

figure(1)
hold on
plot(state_hist(:,1),state_hist(:,2),'b')
plot(state_est_hist(:,1),state_est_hist(:,2),'r')

figure(2)
subplot(3,1,1)
plot(t,state_hist(:,1)-state_est_hist(:,1),'b')
hold on
plot(t,2*sqrt(sigma_hist(:,1)),'r')
plot(t,-2*sqrt(sigma_hist(:,1)),'r')
ylabel('x position error (m)')
subplot(3,1,2)
plot(t,state_hist(:,2)-state_est_hist(:,2),'b')
hold on
plot(t,2*sqrt(sigma_hist(:,2)),'r')
plot(t,-2*sqrt(sigma_hist(:,2)),'r')
ylabel('y position error (m)')
subplot(3,1,3)
plot(t,state_hist(:,3)-state_est_hist(:,3),'b')
hold on
plot(t,2*sqrt(sigma_hist(:,3)),'r')
plot(t,-2*sqrt(sigma_hist(:,3)),'r')
ylabel('Heading error (rad)')

% figure(3)
% subplot(3,2,1)
% plot(t,K_hist(:,1));
% ylabel('1,1')
% subplot(3,2,2)
% plot(t,K_hist(:,2));
% ylabel('1,2')
% subplot(3,2,3)
% plot(t,K_hist(:,3));
% ylabel('2,1')
% subplot(3,2,4)
% plot(t,K_hist(:,4));
% ylabel('2,2')
% subplot(3,2,5)
% plot(t,K_hist(:,5));
% ylabel('3,1')
% subplot(3,2,6)
% plot(t,K_hist(:,6));
% ylabel('3,2')