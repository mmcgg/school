function [ mu, sigma,sigma_bar, K ] = kalman_filter( mu, sigma, u, z, A, B, C, D, R, Q )
%KALMAN_FILTER Summary of this function goes here

    mu_bar = A*mu + B*u;
    sigma_bar = A*sigma*A' + R;
    
    K = sigma_bar*C'*inv(C*sigma_bar*C'+Q);

    mu = mu_bar + K*(z - C*mu_bar);
    sigma = (eye(size(K*C))-K*C)*sigma_bar;


end

