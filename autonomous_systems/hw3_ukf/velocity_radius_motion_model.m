function [ new_state ] = velocity_radius_motion_model(state,input,dt)
    x = state(1);
    y = state(2);
    theta = state(3);
    
    vhat = input(1);
    what = input(2);
    
    x = x - vhat/what*sin(theta) + vhat/what*sin(theta+what*dt);
    y = y + vhat/what*cos(theta) - vhat/what*cos(theta+what*dt);    
    theta = theta + what*dt;
    
    new_state = [x; y; theta];

end

