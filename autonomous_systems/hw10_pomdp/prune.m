function [ x_prune ] = prune( x )
%PRUNE Summary of this function goes here
%   Detailed explanation goes here
    x_prune = x(1,:);
    uniques = 1;
    for i = 2:size(x,1)
        if sum(x(i,:)==x_prune(uniques,:))<3
            x_prune = [x_prune;
                       x(i,:)];
            uniques = uniques + 1;
        end
    end

end

