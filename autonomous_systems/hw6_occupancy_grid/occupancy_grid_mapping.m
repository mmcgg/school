function [ gridmap ] = occupancy_grid_mapping( gridmap, x, z_vec )

    % Iterate over map rows
    for i = 1:size(gridmap,1)
        % Iterate over map columns
        for j = 1:size(gridmap,2)
            % Assume this cell is in the perceptual field of the robot
            gridmap(i,j) = gridmap(i,j) + inverse_sensor_model(i,j,x,z_vec);
        end
    end


end

