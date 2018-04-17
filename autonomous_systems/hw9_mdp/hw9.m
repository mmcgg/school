close all; clear all; clc;

% Get the map variables
MDP_hw_map;
figure(10)

gamma = .995;
pside = .1;
pahead = .8;

%% Make a Value map
V = ones(size(map))*-2;

% reward for exterior walls
V(2,2:N) = -100;
V(2:N+1,2) = -100;
V(N+1,2:N+1) = -100;
V(2:N+1,N+1) = -100;

% reward for obstacles
V(20:40,30:80) = -5000;
V(10:20,60:65) = -5000;

% Another obstacle
V(45:65,10:45) = -5000;

% Another obstacle
V(43:92,75:85) = -5000;
V(70:80,50:75) = -5000;

% reward for goal states
V(75:80,96:98) = 100000;

Vold = V;

%% Do Value Iteration to find the true value function
figure(10)
diff = 1000;
while diff>1e-1
    figure(10)
    surf(V')
    axis([0,102,0,102,0,100000])
    pause(.0001)
    for i=2:size(map,1)-1
        for j=2:size(map,2)-1

            % Don't iterate over the goal, or obstacles or walls
            if map(i,j) ~= 1
                % Get value of northern cell
                n_cell = V(i,j+1);

                % Get value of eastern cell
                e_cell = V(i+1,j);

                % Get value of southern cell
                s_cell = V(i,j-1);

                % Get value of western cell
                w_cell = V(i-1,j);

                % Find the maximum value choice from this cell
                go_north = pside*w_cell + pside*e_cell + pahead*n_cell;
                go_east = pside*n_cell + pside*s_cell + pahead*e_cell;
                go_south = pside*w_cell + pside*e_cell + pahead*s_cell;
                go_west = pside*n_cell + pside*s_cell + pahead*w_cell;

                [max_v,u] = max([go_north, go_east, go_south, go_west]);

                V(i,j) = -2*0 + gamma*max_v;
            end
        end
    end
    % Check for convergence
    diff = norm(V-Vold);
    Vold=V;
end

%% With the new value function, show the optimal path from a start location
x = [50,70];
figure(1)
hold on
progress = 1;
while progress>0
    % Get value of northern cell
    n_cell = V(x(1),x(2)+1);

    % Get value of eastern cell
    e_cell = V(x(1)+1,x(2));

    % Get value of southern cell
    s_cell = V(x(1),x(2)-1);

    % Get value of western cell
    w_cell = V(x(1)-1,x(2));

    % Find the best path
    [max_v,u] = max([n_cell, w_cell, s_cell, e_cell]);
    
    draw_arrow(x(1),x(2),1,(u-1)*pi/2)
    progress = max_v - V(x(1),x(2));
    
    % Move to the new cell
    if(u==1) %Move north
        x(2) = x(2) + 1;
    elseif(u==2) %Move west
        x(1) = x(1) - 1;
    elseif(u==3) %Move south
        x(2) = x(2) - 1;
    elseif(u==4) %Move east
        x(1) = x(1) + 1;
    end
end