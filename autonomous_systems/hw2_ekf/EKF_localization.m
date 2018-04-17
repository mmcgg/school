function [ mu, sigma, K ] = EKF_localization( mu, sigma, u, z, m, dt, alpha, sigma_range, sigma_bearing)

    v = u(1);
    w = u(2);
    theta = mu(3);
    
    % Prediction Step
    G = eye(3);
    G(1,3) = -v/w*cos(theta) + v/w*cos(theta+w*dt);
    G(2,3) = -v/w*sin(theta) + v/w*sin(theta+w*dt);
    
    V = [(-sin(theta)+sin(theta+w*dt))/w, (v*(sin(theta)-sin(theta+w*dt))/w^2 + v*(cos(theta+w*dt)/w*dt));
        (cos(theta)-cos(theta+w*dt))/w, (-v*(cos(theta)-cos(theta+w*dt))/w^2 + v*(sin(theta+w*dt)/w*dt));
        0, dt];
    
    M = [alpha(1)*v^2 + alpha(2)*w^2, 0;
        0, alpha(3)*v^2 + alpha(4)*w^2];
    
    mubar = mu + [-v/w*sin(theta) + v/w*sin(theta+w*dt);
                  v/w*cos(theta) - v/w*cos(theta+w*dt);
                  w*dt];
              
    sigmabar = G*sigma*G' + V*M*V';
    
    % Measurement update step
    Q = [sigma_range^2 0;
        0 sigma_bearing^2];
    
    num_landmarks = size(m);
    for i=1:num_landmarks(1)
        
        q = (m(i,1)-mubar(1))^2 + (m(i,2)-mubar(2))^2;
        zhat = [sqrt(q);
                atan2(m(i,2)-mubar(2),m(i,1)-mubar(1))-mubar(3)];
            
        H = [-(m(i,1)-mubar(1))/sqrt(q) -(m(i,2)-mubar(2))/sqrt(q) 0;
             (m(i,2)-mubar(2))/q -(m(i,1)-mubar(1))/q -1];
         
        S = H*sigmabar*H' + Q;
        K = sigmabar*H'*inv(S);
        mubar = mubar + K*(z(:,i)-zhat);
        sigmabar = (eye(3)-K*H)*sigmabar;
        
    end
    
    mu = mubar;
    sigma = sigmabar;
    

end

