function [ new_state ] = sample_motion_model(state,input,dt, alpha)

    x = state(:,1);
    y = state(:,2);
    theta = state(:,3);

    % The commanded linear and angular velocities
    v = input(1);
    w = input(2);

    % These are the actual linear and angular velocities + noise
    vhat = v + alpha(1)*v^2.*randn(length(x),1) + alpha(2)*w^2.*randn(length(x),1);
    what = w + alpha(3)*v^2.*randn(length(x),1) + alpha(4)*w^2.*randn(length(x),1);
    
    if(length(unique(x))<100)
        x = x - vhat./what.*sin(theta) + vhat./what.*sin(theta+what*dt) + .01*randn(length(x),1);
        y = y + vhat./what.*cos(theta) - vhat./what.*cos(theta+what*dt) + .01*randn(length(x),1);    
        theta = theta + what*dt + .01*randn(length(x),1);
        
    else
        x = x - vhat./what.*sin(theta) + vhat./what.*sin(theta+what*dt);
        y = y + vhat./what.*cos(theta) - vhat./what.*cos(theta+what*dt);    
        theta = theta + what*dt;
    end
    
    new_state = [x, y, theta];

end

