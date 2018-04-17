function [ mu, sigma, K ] = UKF_localization( mu, sigma, u, z, m, dt, alpha, sigma_range, sigma_bearing)

    v = u(1);
    w = u(2);
    theta = mu(3);
    
    ukf_alpha = 1;
    beta = 2; % 2 is optimal for gaussian noise
    kappa = 1;
    
    % Generate augmented mean and covariance
    M = [alpha(1)*v^2 + alpha(2)*w^2, 0;
        0, alpha(3)*v^2 + alpha(4)*w^2];
    
    Q = [sigma_range^2 0;
         0 sigma_bearing^2];
    
    mu_aug = [mu;0;0;0;0];
    
    sigma_aug = [sigma, zeros(3,4);
                 zeros(2,3), M, zeros(2,2);
                 zeros(2,5), Q];
             
             
    % Do sigma point math wizardry (pg. 66 in Probabilistic Robotics)
    n = 7;
    lambda = ukf_alpha^2*(n+kappa)-n;
    gamma = sqrt(n+lambda);
    wc = ones(1,2*n+1);
    wm = ones(1,2*n+1);
    wm(1) = lambda/(n+lambda);
    wc(1) = lambda/(n+lambda) + (1-ukf_alpha^2+beta);
    wm(2:end) = wm(2:end)*1/(2*(n+lambda));
    wc(2:end) = wc(2:end)*1/(2*(n+lambda));
    
    % Generate Sigma Points
    X = [mu_aug, mu_aug + gamma*chol(sigma_aug), mu_aug - gamma*chol(sigma_aug)];
    Xbar = zeros(size(X));
    
    % Pass sigma points through motion model
    for i=1:length(X)
        Xbar(1:3,i) = velocity_radius_motion_model(X(1:3,i),u+X(4:5,i),dt);
    end
    
    % Compute gaussian statistics
    mubar = zeros(3,1);
    mubar(1) = dot(wm,Xbar(1,:));
    mubar(2) = dot(wm,Xbar(2,:));
    mubar(3) = dot(wm,Xbar(3,:));
    
    sigmabar = zeros(3,3);
    for i=1:length(X)
        sigmabar = sigmabar + wc(i)*(Xbar(1:3,i)-mubar)*(Xbar(1:3,i)-mubar)';
    end

    % Predict observations at sigma points
    Zbar = range_bearing_measurement_model(Xbar(1:3,:),m(1,:)) + X(6:7,:);
    
    % Compute gaussian statistics
    zhat = zeros(2,1);
    zhat(1) = dot(wm,Zbar(1,:));
    zhat(2) = dot(wm,Zbar(2,:));
    
    S = zeros(2,2);
    for i=1:length(X)
        S = S + wc(i)*(Zbar(:,i)-zhat)*(Zbar(:,i)-zhat)';
    end
    
    sigma_xz = zeros(3,2);
    for i=1:length(X)
        sigma_xz = sigma_xz + wc(i)*(Xbar(1:3,i)-mubar)*(Zbar(:,i)-zhat)';
    end
    
    % Update mean and covariance
    K = sigma_xz/S;
    mu = mubar + K*(z-zhat);
    sigma = sigmabar - K*S*K';
    
end


