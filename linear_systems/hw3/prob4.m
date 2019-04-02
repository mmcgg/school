syms t t0 t1 'real'

A = [0 0;
     1 0];
B = [1;
     0];
C = [0,1];
D = 0;

eAt = expm(A*t);

Wc = int(eAt*B*B'*eAt',t,t1-t0,0);

Wr = int(eAt*B*B'*eAt',t,0,t0-t1);

Controllability = [B A*B]