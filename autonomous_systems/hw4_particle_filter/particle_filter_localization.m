function [ X, mu] = particle_filter_localization( points, u, z, landmarks, dt, sigma_range, sigma_bearing, alpha)

    [num_points, num_states] = size(points);
  
    % get a sample of states given past states, input, and motion model
    x = sample_motion_model(points,u,dt,alpha);

    % Get a weight for x based on the measurement, state, and measurement
    % model
    w = get_weights(x,z,landmarks, sigma_range, sigma_bearing);
    
    
    scatter(x(:,1),x(:,2),10,w*100)
    axis([-10,10,-10,10])
    pause(.01)
    
    
       
    % Sample from x with weights Xbar
    X = low_variance_sampler(x,w);

    scatter(X(:,1),X(:,2),15,'r')
    axis([-10,10,-10,10])
    pause(.01)

    % Calculate the mean of X to get the state estimate
    mu = mean(X,1);

    scatter(mu(1),mu(2),50,'m*')
    axis([-10,10,-10,10])
    pause(.01)
    
end


