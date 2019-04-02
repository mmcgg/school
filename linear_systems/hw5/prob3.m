A = [1 0 0;
     1 1 0;
     -2 1 1];
 
B = [2;0;0];

C = [1 0 1];

D = [0];

%% Part A
% This is rank 3 => it is observable
rank(obsv(A,C))

%% Part B
syms a b c real
k = [a b c]';
bla = simplify(eig(A+k*C))
simplify(det(-eye(3)-(A+k*C)))

ans = solve(det(-eye(3)-(A+k*C)))

k = [ans.a,ans.b,ans.c]';
k = [1.0,1.0,-15/4]';
%k = [1.0,8.5,0]';
eig(A+k*C)

% Could probably use this somehow...
K = place(A',C',[-1;-1.1;-1.2])
eig(A+K'*C)

% I can get one eigenvalue to -1, but not all...

%% Part C
f=[-9 -74 -24];

eig(A+B*f)

% My controller is special. It takes in anything for y and returns f
% That makes my controller stable.
