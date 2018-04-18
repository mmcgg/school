function [ xdot ] = prob_6_odefunc( t,x )
%PROB_6_ODEFUNC Summary of this function goes here
%   Detailed explanation goes here

    xdot =x;

    xdot(1) = x(2) + x(1)*x(3);
    xdot(2) = -x(1) -x(2) + x(2)*x(3);
    xdot(3) = -x(1)^2 - x(2)^2;

end
