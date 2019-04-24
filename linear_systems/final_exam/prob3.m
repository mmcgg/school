%% Part A
A = [-2 4;
     1 1];
 
eig(A)

%% Part B
syms s t t0;
A = [0 0;
     t 0];
 
ilaplace(inv(s*eye(2)-A))
 
Phi = expm(A*(t-t0))

close all;
x0 = [.001;0];
[t,y] = ode45(@odefunc,[0,1000],x0);

my_y = zeros(size(y));
for i=1:length(t)
    Phi = [1 0;
           .5*t(i)^2 1];
    my_y(i,:) = Phi*x0;
end

plot(t,y)
hold on
plot(t,my_y,'r--')

figure()
plot(t,my_y - y)

function xdot = odefunc(t,x)
    A = [0 0;
         t 0];
    xdot = A*x;
end    