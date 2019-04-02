
n = 5;

for i = 1:100
    A = randn(n,n);
    [vA,eigA] = eig(A);
    [vAT,eigAT] = eig(A');
    if norm(eigAT-eigA) < .01
        vA' - vAT
        vAT
    end
    
end