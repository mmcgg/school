A = [2 0 0; 0 -1 0; 0 0 -1];
B = [1 0; 1 0; 0 1];
C = [1 0 2; 0 -1 0];
D = [1 0; 1 0];

eig(A)
% Poles at s=-1(multiplicity 2), s=2 (multiplicity 1)

% Because it is a minimum realization the transmission zeros are invariant
% zeros. So here are both: 0 and 2
tzero(ss(A,B,C,D))