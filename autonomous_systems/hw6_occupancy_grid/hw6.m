clear all ;close all;clc;
load('state_meas_data.mat')

% Make an empty grid (the map)
gridmap = ones([100,100])*.5;
l0 = log(0.5/(1-.5));

for t=1:size(X,2)
    
    
    % Update the occupancy grid given state and measurement
    % Iterate over map rows
    for i = 1:size(gridmap,1)
        % Iterate over map columns
        for j = 1:size(gridmap,2)
            % Assume this cell is in the perceptual field of the robot
            lij = log(gridmap(i,j)/(1-gridmap(i,j))) + inverse_range_sensor_model(i,j,X(:,t),z(:,:,t),thk) - l0;
            gridmap(i,j) = 1 - 1/(1+exp(lij));
        end
    end
    
    
    
    
    
    
    
end


% t = 1:size(X,2);
% figure(1)
% % Display the robot
% drawRobot(X(1,t),X(2,t),X(3,t),[],[],[],t);

figure(2)
% Display the updated occupancy grid
b = bar3(gridmap);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
colorbar

mymap = zeros(100,100);
for i = 1:size(mymap,1)
    for j = 1:size(mymap,2)
        if(gridmap(i,j)>.5)
            mymap(i,j) = 1;
        end
    end
end

imtool(mymap,[0,1])