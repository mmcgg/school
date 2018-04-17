function [ weights ] = get_weights( x,z, landmarks, sigma_range, sigma_bearing )
    num_landmarks = size(landmarks);
    [num_points,num_states] = size(x);

    weights = ones(num_points,1);

    for i=1:num_landmarks

        q = (landmarks(i,1)-x(:,1)).^2 + (landmarks(i,2)-x(:,2)).^2;

        zhat = [sqrt(q), atan2(landmarks(i,2)-x(:,2),landmarks(i,1)-x(:,1))-x(:,3)];

        %weights = weights.*prob_normal((z(i,1)-zhat(:,1)),sigma_range).*prob_normal((z(i,2)-zhat(:,2)),sigma_bearing);
        weights = weights.*pdf('Normal',(z(i,1)-zhat(:,1)),0,sigma_range).*pdf('Normal',(z(i,2)-zhat(:,2)),0,sigma_bearing);

    end

    


%     weights = ones(num_points,1);
%     
%     for i=1:num_points
% 
%         for j=1:num_landmarks
%             q = sqrt((landmarks(j,1)-x(i,1)).^2 + (landmarks(j,2)-x(i,1)).^2);
% 
%             zhat = [q, atan2(landmarks(j,2)-x(i,2),landmarks(j,1)-x(i,1))-x(i,3)];
%             
%             %this_w = prob_normal(z(j,1)-zhat(1),sigma_range)*prob_normal(z(j,2)-zhat(2),sigma_bearing);
%             this_w = pdf('Normal',z(j,1)-zhat(1),0,sigma_range)*pdf('Normal',z(j,2)-zhat(2),0,sigma_bearing);
%             
%             weights(i) = weights(i)*this_w;
%         end
%     end


    weights = weights/norm(weights);


end

