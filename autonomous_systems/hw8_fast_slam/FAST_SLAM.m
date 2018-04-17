function [ Y, best_particle ] = FAST_SLAM( u, z, dt, Y, alpha, sigma_range, sigma_bearing )

    num_particles = size(Y,1);
    num_landmarks = (size(Y,2) - 3)/6;
    weights = ones(num_particles,num_landmarks)*1/num_particles;
    
    
    for particle = 1:num_particles

        mu = reshape(Y(particle,1:3),3,1);
        v = u(1);
        w = u(2);
        
        % These are the actual linear and angular velocities
        v = v + alpha(1)*v^2*randn() + alpha(2)*w^2*randn();
        w = w + alpha(3)*v^2*randn() + alpha(4)*w^2*randn();

        N = num_landmarks;

        mubar = mu +  [-v/w*sin(mu(3)) + v/w*sin(mu(3)+w*dt);
                          v/w*cos(mu(3)) - v/w*cos(mu(3)+w*dt);
                          w*dt];
                      
        Y(particle,1:3) = mubar';

        Q = blkdiag(sigma_range^2,sigma_bearing^2);

        for i = 1:num_landmarks

            % If we can see this landmark
            if(~isnan(z(1,i)))
    
                sigma = reshape(Y(particle,3+6*i-3:3+6*i),2,2);
                % If this landmark has never been seen before
                if(isnan(Y(particle,3+6*i-5)))
                    % Assign the landmark a place (x,y)
                    Y(particle,3+6*i-5) = Y(particle,1) + z(1,i)*cos(z(2,i)+Y(particle,3));
                    Y(particle,3+6*i-4) = Y(particle,2) + z(1,i)*sin(z(2,i)+Y(particle,3));
                    
                    deltas = [Y(particle,3+6*i-5) - Y(particle,1);
                          Y(particle,3+6*i-4) - Y(particle,2)];
             
                    q = deltas'*deltas;
                    
                    H = 1./q.*[ sqrt(q)*deltas(1), sqrt(q)*deltas(2);
                          -deltas(2), deltas(1)];
                    sigma = inv(H)*Q*inv(H)';
                    
                else

                    deltas = [Y(particle,3+6*i-5) - Y(particle,1);
                              Y(particle,3+6*i-4) - Y(particle,2)];


                    q = deltas'*deltas;

                    % This should be the same as the EKF
                    zhat = [sqrt(q);
                            atan2(deltas(2),deltas(1)) - Y(particle,3)];


                    H = 1./q.*[ sqrt(q)*deltas(1), sqrt(q)*deltas(2);
                              -deltas(2), deltas(1)];

                    Q_meas = H*sigma*H' + Q;  

                    K = sigma*H'*inv(Q_meas);

                    Y(particle,3+6*i-5:3+6*i-4) = Y(particle,3+6*i-5:3+6*i-4) + (K*wrapToPi(z(:,i) - zhat))';

                    sigma = (eye(size(K,1)) - K*H)*sigma;
                    Y(particle,3+6*i-3:3+6*i) = reshape(sigma,1,4);
                    weights(particle,i) = det(2*pi*Q_meas)^-.5 * exp(-.5*wrapToPi(z(:,i) - zhat)'*inv(Q_meas)*wrapToPi(z(:,i) - zhat));
                end                
            end
        end
    end
    % THIS IS NOT GOOD
    %weights = ones(num_particles,1)./num_particles;
    w = log(sum(exp(weights),2));
    w = w./sum(w);
    
    % Use the weights to pick the new particles
    Y = low_variance_sampler(Y,w);
    [max_w, idx] = max(w);
    best_particle = Y(idx,:);
end

