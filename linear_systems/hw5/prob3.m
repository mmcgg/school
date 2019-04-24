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
syms s a b c real
k = [a b c]';
%k = [8 28 -2]';
simplify(det(s*eye(3)-(A+k*C)))

% For three eigenvalues at -1, we want this characteristic equation to be
% (s+1)^3 = s^3 + 3s^2 + 3s + 1

% From the s^2 term
% -a + -c - 3 = 3
% -a + -c =  6

% From the s^1 term
% 3 + 4a - b + 2c = 3
% 4a - b + 2c = 0

% From the s^0 term
% -4a + b - c - 1 = 1
% -4a + b - c = 2

coeffs = [-1 0 -1;
          4 -1 2;
          -4 1 -1];
      
nums = [6 0 2]';

x = inv(coeffs)*nums

eig(A+x*C)



%% Part C
f=[-9 -74 -24];

eig(A+B*f)

% My controller is special. It takes in anything for y and returns f
% That makes my controller stable.

% u = f*xhat
% xdothat = A*xhat + B*f*xhat + k*(yhat-y)
% xdothat = A*xhat + B*f*xhat + k*C*xhat-k*y
% xdothat = (A + B*f + k*C)*xhat - k*y
