T = [0 1 0 0;
     -1 0 0 0;
     0 0 1 0;
     0 0 0 1];
 
trplot(T)
tranimate(T)

T = rt2tr(rotx(pi/4)*roty(pi/4)*rotz(pi/4), [1,1,1]);
plot_cube(T);