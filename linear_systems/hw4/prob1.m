syms w real

A = [0 1 0 0;
    3*w^2 0 0 2*w;
    0 0 0 1;
    0 -2*w 0 1];

B = [0 0;
    1 0;
    0 0;
    0 1];


%% Part A - is the system controllable?

controllability = [B A*B A*A*B A*A*A*B];
rank(controllability)

% The rank = 4, so the system is fully controllable


%% Part B - is the system controllable with either input alone?

controllability = [B(:,1) A*B(:,1) A*A*B(:,1) A*A*A*B(:,1)];
rank(controllability)

% The rank = 3, so the system is not controllable without the tangential
% thruster

controllability = [B(:,2) A*B(:,2) A*A*B(:,2) A*A*A*B(:,2)];
rank(controllability)

% The rank = 4, so the system is controllable without the radial thruster