function plot_cube(T)

%defining the original vertices of a 2 x 2 x 2 square. I started looking
%for a pattern to define this to be more general, but then just finished
%picking off the needed vertices by inspection :).
original_vert = [1, 1, 1;
    1, -1, 1;
    -1, -1, 1;
    -1, 1, 1;
    1, 1, 1;
    1, 1, -1;
    1, -1, -1;
    -1, -1, -1;
    -1, 1, -1;
    1, 1, -1;
    1, -1, -1;
    1, -1, 1;
    -1, -1, 1;
    -1, -1, -1;
    -1, 1, -1;
    -1, 1, 1;];

%getting size that will be needed to make it a homogeneous coordinate point
[r,c] = size(original_vert);

%drawing original square
figure();
plot3(original_vert(:, 1), original_vert(:, 2), original_vert(:, 3));
xlabel('x')
ylabel('y')
zlabel('z')
axis equal;

%it would be better if the number of steps (100) was variable or depended
%on the total rotation/translation
num_steps = 100
for i = 1:1:num_steps
    %taking a small step of the total translation and rotation
    T_next = trinterp(T, i/num_steps);
    
    %calculating the new location of each vertice
    new_vert = (T_next*[original_vert'; ones(1, r)])';
    
    %plotting the new vertices, along with the origin in red for reference
    plot3(new_vert(:, 1), new_vert(:, 2), new_vert(:, 3), 'b', 0, 0, 0, 'r*');
    xlabel('x')
    ylabel('y')
    zlabel('z')
    axis equal;
    
    %pausing at a frame rate of about 30 Hz.
    pause(1/30.0);
end
             

end