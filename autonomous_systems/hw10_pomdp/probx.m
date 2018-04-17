function [ prob_x_prime ] = probx( x_prime,u,x )
    
    if u==3
        if x_prime==x
            prob_x_prime = .2;
        else
            prob_x_prime = .8;
        end
    else
        prob_x_prime = 0;
    end
end

