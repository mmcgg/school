function [ mu, sigma ] = EKF_SLAM( mu, sigma, u, z, dt, num_landmarks, alpha, sigma_range, sigma_bearing )
    v = u(1);
    w = u(2);
    
    N = num_landmarks;
    
    F = [eye(3), zeros(3,2*N)];
    
    mubar = mu + F'* [-v/w*sin(mu(3)) + v/w*sin(mu(3)+w*dt);
                      v/w*cos(mu(3)) - v/w*cos(mu(3)+w*dt);
                      w*dt];

    G = eye(2*N+3) + F'*[0, 0, -v/w*cos(mu(3)) + v/w*cos(mu(3)+w*dt);
                         0, 0, -v/w*sin(mu(3)) + v/w*sin(mu(3)+w*dt);
                         0, 0, 0]*F;
                     
    V = [(-sin(mu(3))+sin(mu(3)+w*dt))/w, (v*(sin(mu(3))-sin(mu(3)+w*dt))/w^2 + v*(cos(mu(3)+w*dt)/w*dt));
        (cos(mu(3))-cos(mu(3)+w*dt))/w, (-v*(cos(mu(3))-cos(mu(3)+w*dt))/w^2 + v*(sin(mu(3)+w*dt)/w*dt));
        0, dt];
    
    M = [alpha(1)*v^2 + alpha(2)*w^2, 0;
        0, alpha(3)*v^2 + alpha(4)*w^2];
                     
    R = V*M*V';
                     
    sigmabar = G*sigma*G' + F'*R*F;
    
    Q = blkdiag(sigma_range^2,sigma_bearing^2);
    
    for i = 1:num_landmarks
        
        % If we can see this landmark
        if(~isnan(z(1,i)))
        
            % If this landmark has never been seen before
            if(isnan(mubar(3+(i-1)*2+1)))
                mubar(3+(i-1)*2+1) = mubar(1) + z(1,i)*cos(z(2,i)+mubar(3));
                mubar(3+(i-1)*2+2) = mubar(2) + z(1,i)*sin(z(2,i)+mubar(3));
            end

            deltas = [mubar(3+(i-1)*2+1)-mubar(1);
                      mubar(3+(i-1)*2+2)-mubar(2)];

            q = deltas'*deltas;

            zhat = [sqrt(q);
                    atan2(deltas(2),deltas(1)) - mubar(3)];

            Fxj = [[eye(3), zeros(3,2*N)];
                   [zeros(2,3), zeros(2,2*i-2), eye(2), zeros(2,2*N - 2*i)]];

            H = 1./q.*[-sqrt(q)*deltas(1), -sqrt(q)*deltas(2) 0, sqrt(q)*deltas(1), sqrt(q)*deltas(2);
                      deltas(2), -deltas(1), -q, -deltas(2), deltas(1)]*Fxj;

            K = sigmabar*H'*inv(H*sigmabar*H' + Q);

            mubar = mubar + K*wrapToPi(z(:,i) - zhat);

            sigmabar = (eye(size(K,1)) - K*H)*sigmabar;
        end
    end
    
    mu = mubar;
    sigma = sigmabar;
    
    
end

