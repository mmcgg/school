A = -eye(2);
B = eye(2);
C = [-1,1];
D = [2,1];

sys = ss(A,B,C,D);

msys = minreal(sys);

msys.A
msys.B
msys.C
msys.D

% The original state space representation was not minimal. The new one is.