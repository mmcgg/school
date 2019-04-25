%% Different cost function LQR
% On paper I found that the only difference in LQR with the new cost
% function is that the algebraic ricatti equation is now with a different A
% matrix which includes I*beta along the diagonal
n = 2;
m = 2;

for i=1:100
    beta = randn()*10;
    A = randn(n,n);
    B = randn(n,m);
    Q = eye(n);
    R = eye(m);

    P = care(A,B,Q,R);
    newA = A+beta*eye(n);
    newP = care(A,B,Q,R);

    disp(norm(P-newP))
end

% This showed me that adding a constant along the diagonal of A does not
% change the solution to the A.R.E

