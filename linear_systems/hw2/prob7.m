% Investigate wheter or not the solutions to the following 
% nonlinear systems convege to the given equilibrium point
% when they start sufficiently close to it:
%% Part A
x0 = .01*[1 10];
[t,y] = ode45(@system1,[0,10],x0);

figure(1)
subplot(2,1,1)
plot(t,y(:,1))
subplot(2,1,2)
plot(t,y(:,2))

% Yes - this converges to [0,0] if started sufficiently close

%% Part B
%syms g
g = -0.01;
A = [g,-1;
     1 0];

d = eig(A)






function [xdot] = system1(t,x)
    xdot = zeros(2,1);
    xdot(1) = -x(1) + x(1)*(x(1)^2 + x(2)^2);
    xdot(2) = -x(2) + x(2)*(x(1)^2 + x(2)^2);
end
