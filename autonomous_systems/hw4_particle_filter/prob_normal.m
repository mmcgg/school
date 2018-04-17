function [ p ] = prob_normal( a,b2 )
    
    p = 1./sqrt(2.*pi.*b2).*exp(-.5*a.^2./b2);


end

