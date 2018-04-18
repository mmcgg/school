close all; clear all; clc;

figure()
hold on

for i=-5:.5:5
    for j=-5:.5:5

        x0 = [i,j];
        tspan = [0,10];
        [t,x] = ode45(@prob1_ode_func,tspan,x0);
        scatter(x(:,1),x(:,2),1)

        
    end
end

x0 = [0,.1];
tspan = [0,1000];
[t,x] = ode45(@prob1_ode_func,tspan,x0);
        
figure()
plot(t,x)
legend('x1','x2')

mu = -2;
a = [mu -1;
     1 0];
[v,d] = eig(a)
