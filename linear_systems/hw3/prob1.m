syms s

A = diag([-2,1,-1]);
B = [1 0 -1]';
C = [1 1 0];
D = 1;

Gs = C*inv(s*eye(3) - A)*B + D

eig(A)

