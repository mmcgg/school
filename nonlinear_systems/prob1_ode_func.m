function [ xdot ] = prob1_ode_func( t,x )
%PROB1_ODE_FUNC Summary of this function goes here
%   Detailed explanation goes here
    mu = 2;
    
    xdot = [0;0];
    
    xdot(1) = -(x(1)*x(2)^2 + x(1)^3 - mu*x(1)) - x(2);
    xdot(2) = x(1);

end
