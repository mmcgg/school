A = [0 -1 1;
     1 -2 1;
     0 1 -1];
 
B = [1 0;
     1 1;
     1 2];
 
C = [0 1 0];

% Is it controllable


rank([B A*B A*A*B])

% This is not rank 3, so it is not controllable

% I check to see which mode is uncontrollable
[v,d] = eig(A);

rank([A - eye(3)*d(1,1), B])
rank([A - eye(3)*d(2,2), B])
rank([A - eye(3)*d(3,3), B])

% The mode corresponding to d(1,1) gives rank 2 instead of 3, so that is
% the uncontrollable mode. However, that eigenvalue is negative, so it is
% stabilizable.

% Is it observable

rank([C; C*A; C*A*A])

% I check to see which mode is unobservable

rank([A - eye(3)*d(1,1); C])
rank([A - eye(3)*d(2,2); C])
rank([A - eye(3)*d(3,3); C])

% The mode corresponding to d(2,2) gives rank 2 instead of 3, so that is
% the unobservable mode. However, that eigenvalue is negative, so it is
% detectable.