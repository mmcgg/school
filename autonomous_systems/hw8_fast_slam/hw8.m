close all; clear all; clc;
%% Implementing the velocity motion model
alpha = [.1, .01, .1, .01];
sigma_range = .1;
sigma_bearing = .05;

dt = .05;
x = 0;
y = 0;
theta = 0*pi/180;
mu = [x;y;theta];
         
landmarks = rand(15,2)*50 -[25, 25];

num_landmarks = size(landmarks,1);

% The structure for each particle is (mean and covariance for robot state,
% then mean and covariance for each landmark)
num_particles = 50;

one_landmark_appendage = [nan(1,2), reshape(eye(2),1,4)*1e10];
my_appendage = [x, y, theta];
for i=1:num_landmarks
    my_appendage = [my_appendage, one_landmark_appendage];
end
Y = [];
for i=1:num_particles
    Y = [Y;
        my_appendage];
end
         
range = zeros([num_landmarks,1]);
bearing = zeros([num_landmarks,1]);

t = 0:dt:30;

v_hist = [];
w_hist = [];
state_hist = [];
state_est_hist = [];
sigma_hist = [];
range_hist = [];
bearing_hist = [];
K_hist = [];


sigma = eye(3+num_landmarks*2)*1e10;
sigma(1,1) = 0;
sigma(2,2) = 0;
sigma(3,3) = 0;
mu = [mu; nan(num_landmarks*2,1)];

figure(1)

for i=1:length(t)
    
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
    for j=1:num_landmarks
        range_reading = landmarks(j,:) - [x,y] + sigma_range*randn();
        range(j) = norm(range_reading);
        bearing(j) = atan2(range_reading(2),range_reading(1)) - theta + sigma_bearing*randn();
        % Adjust the "Field of view"
        if(abs(wrapToPi(bearing(j))) > 1*pi/4)
            bearing(j) = nan;
            range(j) = nan;
        end
    end
    
    u = [v,w];
    z = [range';bearing'];
    
    range_hist = [range_hist; range'];
    bearing_hist = [bearing_hist; bearing'];
    
    % Get an estimate of state using FAST_SLAM
    [Y, best_particle] = FAST_SLAM(u, z, dt, Y, alpha,sigma_range, sigma_bearing);
    
    avgY = nanmean(Y);
    mu = avgY(1:3)';
    sigmax = std(Y(:,1))^2;
    sigmay = std(Y(:,2))^2;
    sigmatheta = std(Y(:,3))^2;
    
    
    
    state_est_hist = [state_est_hist; mu'];
    sigma_hist = [sigma_hist; sigmax sigmay sigmatheta];
    
    % Draw the estimated robot
    clf;
    drawRobot(mu(1),mu(2),mu(3),[],'r')
    hold on
    scatter(Y(:,1),Y(:,2),3,'g')
    
%     plot_covariance([mu(1),mu(2)],sigma(1:2,1:2))
     % Draw the estimated landmarks
    for j =1:num_landmarks
%         if(~isnan(best_particle(3+6*j-5)))
%             plot(best_particle(3+6*j-5), best_particle(3+6*j-4),'rx')
%             plot_covariance([best_particle(3+6*j-5), best_particle(3+6*j-4)],reshape(best_particle(3+6*j-3:3+6*j),2,2))
%         end
        if(~isnan(avgY(3+6*j-5)))
            plot(avgY(3+6*j-5), avgY(3+6*j-4),'rx')
            plot_covariance([avgY(3+6*j-5), avgY(3+6*j-4)],reshape(avgY(3+6*j-3:3+6*j),2,2))
        end
    end
    
    % Draw the real robot and landmarks
    drawRobot(x,y,theta,landmarks,'b')
    
    figure(1)
    axis equal
    pause(.00001)
    
end



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
