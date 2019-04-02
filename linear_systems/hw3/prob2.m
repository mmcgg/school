syms s w

A = diag([-w^2]);
B = [1]';
C = [1];

G1s = C*inv(s*eye(1) - A)*B;

A = [[-w, 1],
     [0, -w]];
B = [0 1]';
C = [1 0];

G2s = C*inv(s*eye(2) - A)*B

eig(A)