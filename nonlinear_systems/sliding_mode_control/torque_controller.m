function u = torque_controller(x)
    Kp = [10, 0;
          0,  5];
      
    
    Kd = [2, 0;
          0,  1];   
    
    qdes = x(1:2);
    qd = x(3:4);
    q = x(5:6);
    
    u = Kp*(qdes-q) - Kd*qd;
end
