close all; clear all; clc;

Yup = [0, 0, 0];
num_actions = 3;
num_meas = 2;
num_states = 2;
T = 2;
gamma = 1.0;

for tau = 1:T
    v = zeros(length(Yup),num_actions,num_meas,num_states);
    
    Yup_prime = [];
    for k = 1:size(Yup,1)
        for u = 1:num_actions
            for z = 1:num_meas
                for j = 1:num_states
                    
                    for i = 1:num_states
                        v(k,u,z,j) = v(k,u,z,j) + Yup(k,i+1) * probz(z,i) * probx(i,u,j);
                    end
                    
                end
            end
        end
    end
    
    for u = 1:num_actions
        for k1 = 1:length(Yup)
            for k2 = 1:length(Yup)
                v_prime = zeros(1,num_states);
                for i = 1:num_states
                    v_prime(i) = gamma*(reward(i,u) + v(k1,u,1,i) + v(k2,u,2,i));
                end
                
                % Add v_prime to Yup prime
                Yup_prime = [Yup_prime;
                            [u,v_prime]];
                        
            end
        end
    end
    
    % Prune Yup_prime
    Yup_prime = prune(Yup_prime);
    Yup = Yup_prime;
    
    figure(tau)
    hold on
    for iter=1:size(Yup,1)
        if Yup(iter,1)==1
            plot([1,0],Yup(iter,2:3),'r')
        elseif Yup(iter,1)==2
            plot([1,0],Yup(iter,2:3),'b')
        elseif Yup(iter,1)==3
            plot([1,0],Yup(iter,2:3),'k')
        end
    end
    pause(.01);
end