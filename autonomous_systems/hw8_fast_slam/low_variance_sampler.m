function [ Xbar ] = low_variance_sampler( X,W )

    [num_points, num_states] = size(X);
    
    Xbar =[];
    
    r = rand()/num_points;
    c = W(1);
    i = 1;
    
    for m=1:num_points
        
        U = r + (m-1)/num_points;
        while U > c
            i = i + 1;
            c = c + W(i);
        end
        Xbar = [Xbar; X(i,:)];
        
    end


end

