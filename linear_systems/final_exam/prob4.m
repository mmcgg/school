load('sat.mat')

D1 = .001*eye(3);
N_theta = 1e-5*eye(3);
N_w = 1e-10*eye(3);

D = B*D1*B';
N = blkdiag(N_theta,N_w);

%% Parts A and B - make an LQR controller and Kalman Filter
close all; 

% Design LQR Controller
Q = eye(18);
R = eye(3);
K = lqr(A,B,Q,R);

% Design Kalman Filter
G = eye(size(A,1));
L = lqe(A,G,C,D,N);

x = ones(18,1);
xhat = ones(18,1);
xdes = zeros(18,1);

dt = .00005;
t_final = 200;

xhist = zeros(18,t_final/dt+1);
uhist = zeros(3,t_final/dt+1);
xhathist = zeros(18,t_final/dt+1);
ehist = zeros(18,t_final/dt+1);

for t=0:dt:t_final

    % Calculate Control using estimated state and LQR
    u = K*(xdes-xhat);
    
    % Saturate control
    u = sign(u).*min(ones(3,1),abs(u));
    
    % Forward simulate states and outputs
    y = C*x + N*randn(6,1);
    xdot = A*x + B*u;
    x = x + xdot*dt + D*randn(18,1);

    % Update estimates
    yhat = C*xhat;
    xhatdot = A*xhat + B*u - (L*(yhat-y));
    xhat = xhat + xhatdot*dt;

    % Record stuff for plotting
    idx = round((t+dt)/dt);
    xhist(:,idx) = x;
    uhist(:,idx) = u;
    xhathist(:,idx) = xhat;
    ehist(:,idx) = x-xhat;    
end

% Plot
t=0:dt:t_final;
figure(1)
plot(t,xhist(1,:),'r')
hold on
plot(t,xhist(2,:),'g')
plot(t,xhist(3,:),'b')
plot(t,xhist(10,:),'r')
plot(t,xhist(11,:),'g')
plot(t,xhist(12,:),'b')
plot(t,xhathist(1,:),'r--')
plot(t,xhathist(2,:),'g--')
plot(t,xhathist(3,:),'b--')
plot(t,xhathist(10,:),'r--')
plot(t,xhathist(11,:),'g--')
plot(t,xhathist(12,:),'b--')
title('Position and Velocity states')

figure(2)
plot(t,xhist(4:9,:))
hold on
plot(t,xhist(13:18,:))
legend('4','5','6','7','8','9','13','14','15','16','17','18')
title('Mystery states')

figure(3)
plot(t,uhist)
title('inputs')

figure(4)
plot(t,ehist)
title('Estimation Error')
%% Part C - Use LQG to drive to a nonzero point

x = zeros(18,1);
xdes = zeros(18,1);
xdes(1) = 10;

xhat = x;
xhist = zeros(18,t_final/dt+1);
uhist = zeros(3,t_final/dt+1);
xhathist = zeros(18,t_final/dt+1);
ehist = zeros(18,t_final/dt+1);

for t=0:dt:t_final

    % Calculate Control using estimated state and LQR
    u = K*(xdes-xhat);
    
    % Saturate control
    u = sign(u).*min(ones(3,1),abs(u));
    
    % Forward simulate states and outputs
    y = C*x + N*randn(6,1);
    xdot = A*x + B*u;
    x = x + xdot*dt + D*randn(18,1);

    % Update estimates
    yhat = C*xhat;
    xhatdot = A*xhat + B*u - (L*(yhat-y));
    xhat = xhat + xhatdot*dt;

    % Record stuff for plotting
    idx = round((t+dt)/dt);
    xhist(:,idx) = x;
    uhist(:,idx) = u;
    xhathist(:,idx) = xhat;
    ehist(:,idx) = x-xhat;    
end

% Plot
t=0:dt:t_final;
figure(5)
plot(t,xhist(1,:),'r')
hold on
plot(t,xhist(2,:),'g')
plot(t,xhist(3,:),'b')
plot(t,xhist(10,:),'r')
plot(t,xhist(11,:),'g')
plot(t,xhist(12,:),'b')
plot(t,xhathist(1,:),'r--')
plot(t,xhathist(2,:),'g--')
plot(t,xhathist(3,:),'b--')
plot(t,xhathist(10,:),'r--')
plot(t,xhathist(11,:),'g--')
plot(t,xhathist(12,:),'b--')
title('Position and Velocity states')

figure(6)
plot(t,xhist(4:9,:))
hold on
plot(t,xhist(13:18,:))
legend('4','5','6','7','8','9','13','14','15','16','17','18')
title('Mystery states')

figure(7)
plot(t,uhist)
title('inputs')

figure(8)
plot(t,ehist)
title('Estimation Error')