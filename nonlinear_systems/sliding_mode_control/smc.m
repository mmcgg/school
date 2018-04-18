
function u = smc(x)
    global twolink
    

    Kp = [10, 0;
          0,  5];
    Kd = [2, 0;
          0,  1];
      
    a1 = 4;
    beta0 = .1;

    qd = x(1:2);
    q = x(3:4);
    
    u = [0; 0];
    
    M = twolink.inertia(q');
    C = twolink.coriolis(qd',q');
    
    A = [-inv(M)*(C+Kd), -inv(M)*Kp;
         eye(2), zeros(2,2)];
     
    f = A*x;
    f = f(1:2);
    rho = inv(inv(M)*Kp)*(a1*qd + f);
    s = a1*q + qd
    
    for i=1:2
        if(abs(s(i))>0.01)
            u(i) = -(abs(rho(i)) + beta0)*sign(s(i));
        else
            u(i) = -(abs(rho(i)) + beta0)*s(i);
        end
        %u(i) = min(0.1, max(-0.1, u(i)))
    end
    
    disp("u");
    disp(u);
    
        


end

