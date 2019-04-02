A = [-.038, 18.984, 0, -32.174;
     -.001, -.632, 1, 0;
     0, -.759, -.518, 0;
     0, 0, 1, 0];
 
B = [10.1, 0;
     0, -.0086;
     .025, -.011;
     0, 0];
 
C = eye(4);
D = [];

sys = ss(A,B,C,D);

%% Compute the modes

% The modes are the eigenvectors of A
[v,d] = eig(A);
v


inv(v)*B
% It looks like 