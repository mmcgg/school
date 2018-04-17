close all; clear all; clc;
%% Implementing the velocity motion model
alpha = [.1, .01, .01, .1];
sigma_range = .1;
sigma_bearing = .05;

dt = .1;
x = -5;
y = -3;
theta = 90*pi/180;

num_points = 1000;

landmarks =   [-7 8;
               6 4;
               6 -4];
             
landmarks = randn(10,2)*10;
         
num_landmarks = size(landmarks);
         
range = zeros([num_landmarks(1),1]);
bearing = zeros([num_landmarks(1),1]);

t = 0:dt:20;

v_hist = [];
w_hist = [];
state_hist = [];
state_est_hist = [];
sigma_hist = [];
range_hist = [];
bearing_hist = [];
K_hist = [];

mu = [x;y;theta];

figure(1)

% Create a set of random uniformly distributed particles
points = rand(num_points,3);
points(:,1:2) = points(:,1:2)*20 - 10;
points(:,3) = points(:,3)*2*pi;
%points(:,3) = pi/2;

for i=1:length(t)

    clf
    scatter(points(:,1),points(:,2),5,'g')
    hold on
    scatter(x,y,50,'r')
    axis([-10,10,-10,10])
    pause(.01)
    
    % Calculate the commanded linear and angular velocities
    v = 1 + .5*cos(2*pi*(.2*t(i)));
    w = -.2 + 2.0*cos(2*pi*(.6*t(i)));
    
    % These are the actual linear and angular velocities
    vhat = v + alpha(1)*v^2*randn() + alpha(2)*w^2*randn();
    what = w + alpha(3)*v^2*randn() + alpha(4)*w^2*randn();
    
    v_hist = [v_hist;v];
    w_hist = [w_hist;w];
    
    % Integrate the velocities to get position and heading
    % Radius motion model (from pg 126 in Probabilistic Robotics book)
    x = x - vhat/what*sin(theta) + vhat/what*sin(theta+what*dt);
    y = y + vhat/what*cos(theta) - vhat/what*cos(theta+what*dt);    
    theta = theta + what*dt;    
    state_hist = [state_hist; x y theta];
    
    % Get measurements from landmarks (range and bearing)
    for j=1:num_landmarks(1)
        range_reading = landmarks(j,:) - [x,y] + sigma_range*randn()*0;
        range(j,:) = norm(range_reading);
        bearing(j,:) = atan2(range_reading(2),range_reading(1)) - theta + sigma_bearing*randn()*0;
    end
    
    u = [v;w];
    z = [range';bearing'];
        
    range_hist = [range_hist; range'];
    bearing_hist = [bearing_hist; bearing'];
    
    % Get an estimate of state using a particle filter
    [points,mu] = particle_filter_localization(points,u,z',landmarks,dt,sigma_range,sigma_bearing,alpha);
    
    state_est_hist = [state_est_hist; mu];
    
    
    %waitforbuttonpress;
    
end

% Draw the robot
%drawRobot(state_hist(:,1),state_hist(:,2),state_hist(:,3),landmarks,range_hist,bearing_hist,t)

figure(1)
hold on
plot(state_hist(:,1),state_hist(:,2),'b')
plot(state_est_hist(:,1),state_est_hist(:,2),'r')

figure(2)
subplot(3,1,1)
plot(t,state_hist(:,1)-state_est_hist(:,1),'b')
hold on
%plot(t,2*sqrt(sigma_hist(:,1)),'r')
%plot(t,-2*sqrt(sigma_hist(:,1)),'r')
ylabel('x position error (m)')
subplot(3,1,2)
plot(t,state_hist(:,2)-state_est_hist(:,2),'b')
hold on
%plot(t,2*sqrt(sigma_hist(:,2)),'r')
%plot(t,-2*sqrt(sigma_hist(:,2)),'r')
ylabel('y position error (m)')
subplot(3,1,3)
plot(t,state_hist(:,3)-state_est_hist(:,3),'b')
hold on
%plot(t,2*sqrt(sigma_hist(:,3)),'r')
%plot(t,-2*sqrt(sigma_hist(:,3)),'r')
ylabel('Heading error (rad)')