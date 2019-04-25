%% Part D
A = [0 1;
     2 0];
 
B1 = [0;2];
B2 = [0;-2];

C = [1 0];

% Is it stable?
eig(A)
% No - there is a positive eigenvalue in A

%% Part E
rank(ctrb(A,B1))
rank(ctrb(A,B2))
rank(obsv(A,C))