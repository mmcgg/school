close all; clear all; clc;

figure(1)
hold on

figure(2)
hold on

figure(3)
hold on

for i=-2:2
    for j=-2:2
        for k=-5:5        

            x0 = [i,j,k];
            tspan = [0,10];
            [t,x] = ode45(@prob_6_odefunc,tspan,x0);
            
            figure(1)
            scatter(x(:,1),x(:,2),1)
            
            figure(2)
            scatter(x(:,1),x(:,3),1)
            
            figure(3)
            plot3(x(:,1),x(:,2),x(:,3))

        end        
    end
end

xlabel('X')
ylabel('Y')
zlabel('Z')

x0 = [1,1,1];
tspan = [0,1000];
[t,x] = ode45(@prob_6_odefunc,tspan,x0);
        
figure()
plot3(x(:,1),x(:,2),x(:,3))
