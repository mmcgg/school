load('sat.mat')

% [msys,U] = minreal(ss(A,B,C,D));
% A = msys.A;
% B = msys.B;
% C = msys.C;
% D = msys.D;


D1 = .001*eye(3);
N_theta = 1e-5*eye(3);
N_w = 1e-10*eye(3);

D = B*D1*B';
N = blkdiag(N_theta,N_w);

%% Part A - make an LQR controller and Kalman Filter
close all; 
Q = diag([1*ones(1,3)/pi, 0*ones(1,6), 10*ones(1,3)/(3*pi), 0*ones(1,6)]);
R = 10*diag([1,1,1]);
K = lqr(A,B,Q,R);

D = diag([1e-10*ones(1,3)/pi, 1e-1*ones(1,6), 1e-10*ones(1,3), 1e-1*ones(1,6)]);
N = 1e-10*eye(6);
G = eye(size(A,1));
L = lqe(A,G,C,D,N);

x = 1*ones(size(A,1),1);
xdes = 0*ones(size(A,1),1);
xdes(1) = -1;
xhat = -0.5*ones(size(A,1),1);
xhist = zeros(size(A,1),t_final/dt+1);
uhist = zeros(size(B,2),t_final/dt+1);
xhathist = zeros(size(A,1),t_final/dt+1);
ehist = zeros(size(A,1),t_final/dt+1);
dt = .001;
t_final = 150;
for t=0:dt:t_final

    idx = round((t+dt)/dt);
    u = K*(xdes-xhat);
    %u = K*(xdes-x);
    u = sign(u).*min(ones(3,1),abs(u));
    
    y = C*x;
    xdot = A*x + B*u;
    x = x + xdot*dt;

    xhist(:,idx) = x;
    uhist(:,idx) = u;

    yhat = C*xhat;
    xhatdot = A*xhat + B*u - (L*(yhat-y));
    xhat = xhat + xhatdot*dt;

    xhathist(:,idx) = xhat;
    ehist(:,idx) = x-xhat;    
end

t=0:dt:t_final;
figure(5)
%plot(t,xhist)
plot(t,xhist(1:3,:))
hold on
plot(t,xhist(10:12,:))
plot(t,xhathist(1:3,:),'r')
plot(t,xhathist(10:12,:),'r')
title('Position and Velocity states')
legend('1','2','3','10','11','12')

figure(6)
plot(t,xhist(4:9,:))
hold on
plot(t,xhist(13:18,:))
legend('4','5','6','7','8','9','13','14','15','16','17','18')
title('Mystery states')

figure(7)
plot(t,uhist)
title('inputs')

% figure(8)
% plot(t,ehist)
% title('Estimation Error')