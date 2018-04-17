function [ logodds ] = inverse_range_sensor_model( map_x,map_y,state,z,thk )
%INVERSE_RANGE_SENSOR_MODEL Summary of this function goes here
%   Detailed explanation goes here
    alpha = 1; %m
    beta = 5*pi/180; %rad
    z_max = 150; %m
    l_0 = log(0.5/(1-.5));
    l_free = log(0.3/(1-.3));
    l_occ = log(0.7/(1-.7));    
    
    r = norm([(map_x-state(1)),(map_y-state(2))]);
    
    phi = atan2(map_y-state(2),map_x-state(1)) - state(3); %Fix wrapping here
    [mymin,k] = min(abs(phi-thk));
    
    if(isnan(z(1,k)))
        z(1,k) = 150;
    end
    
    % If this cell is not within the perceptual field of measurement k
    if r>min(z_max,z(1,k)) || abs(phi-thk(k))>beta/2
        logodds = l_0;
    
    % If this cell is occupied
    elseif z(1,k) < z_max && abs(r-z(1,k))<alpha/2
        logodds = l_occ;
    
    % If this cell is free
    elseif r < z(1,k)
        logodds = l_free;
        
    else
        
        disp("Didn't get in any of my else things");
        logodds = l_free;
    end

end

