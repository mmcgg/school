function [ r ] = reward( x,u )
    
    if u==1
        if x==1
            r = -100;
        elseif x==2
            r = 100;
        end
        
    elseif u==2
        if x==1
            r = 100;
        elseif x==2
            r = -50;
        end
    
    elseif u==3
        r = -1;
    end

end

