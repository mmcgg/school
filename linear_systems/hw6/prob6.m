A = [-1 0 0; 0 -2 0; 0 0 -2];
B = [2 -2; -2 4; -4 2];
C = [1 1 0; 1 0 1];
D = [0 0; 0 0];

zero = tzero(ss(A,B,C,D));

syms s
G = C*inv(s*eye(3) - A)*B + D;
P = [(s*eye(3)-A) B; -C D];

P1 = subs(P,s,zero);

xu0 = eval(null(P1));

x0 = xu0(1:3);
u0 = xu0(4:5);

t = linspace(0,10);
u = exp(zero*t);
input = u0*u;

[y,x] = lsim(ss(A,B,C,D),input,t,x0);
plot(t,y)



