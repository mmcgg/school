twolink = SerialLink([
    Revolute('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [-0.5 0 0], 'I', eye(3) , 'B', 1000 , 'G', 0, 'Jm', 0, 'standard'),
    Revolute('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [-0.5 0 0], 'I', eye(3) , 'B', 1000 , 'G', 0, 'Jm', 0, 'standard')
    ], ...
    'name', 'two link');



Kp = [10, 0;
          0,  5];
Kd = [2, 0;
      0,  1];

qd = [0, 0];
q = [0, 0];
dt = .01;

M = twolink.inertia(q)
C = twolink.coriolis(qd,q); % qd, q or q, qd
A = [-inv(M)*(C+Kd), -inv(M)*Kp;
     eye(2), zeros(2,2)];
B = [inv(M)*Kp; zeros(2,2)];
C = eye(4);
D = zeros(4,2);
sys = ss(A,B,C,D);
%sysd = c2d(sys,dt,'zoh');
sysd = c2d(sys,dt,'foh');
%sysd = c2d(sys,dt,'impulse');
%sysd = c2d(sys,dt,'tustin');
Ad = sysd.A;
Bd = sysd.B

disp('first input state weightings')
abs(Bd(:,1)')/norm(Bd(:,1))

disp('second input state weightings')
abs(Bd(:,2))'/norm(Bd(:,2))

my_Ad = inv(eye(size(A)) - A*dt)
my_Bd = inv(eye(size(A)) - A*dt)*B*dt
