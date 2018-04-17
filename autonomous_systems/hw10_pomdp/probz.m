function [ prob_z ] = probz( z,x )

    if z==x
        prob_z = .7;
    else
        prob_z = .3;
    end

end

