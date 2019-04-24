A = [-4, -2;
     0, 1];
 
B = [1;0];

C = [-2 1];
D = 1;

sys = ss(A,B,C,D);

z = tzero(sys);
u = 2;

y = C*(A*z+B*u) + D*u

x0 = [1/3 -1/3]';
u0 = 1;

y = C*(A*x0+B*u0) + D*u0