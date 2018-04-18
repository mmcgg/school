%MDL_ONELINK Create model of a 1-link mechanism
%
%   theta d a alpha
global twolink
twolink = SerialLink([
    Revolute('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [-0.5 0 0], 'I', eye(3) , 'B', 1000 , 'G', 0, 'Jm', 0, 'standard'),
    Revolute('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [-0.5 0 0], 'I', eye(3) , 'B', 1000 , 'G', 0, 'Jm', 0, 'standard')
    ], ...
    'name', 'two link');
