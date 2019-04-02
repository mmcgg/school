A = [1 1 0;
     0 2 0;
     0 1 2];
 
[P,J] = jordan(A)

myP = [1 0 1;
       0 0 1;
       0 1 0];
   
inv(myP)*A*myP