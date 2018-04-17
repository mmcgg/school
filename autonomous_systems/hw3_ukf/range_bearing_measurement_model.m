function [ zhat ] = range_bearing_measurement_model( state, landmark )

    q = (landmark(1)-state(1,:)).^2 + (landmark(2)-state(2,:)).^2;
    zhat = [sqrt(q);
            atan2(landmark(2)-state(2,:),landmark(1)-state(1,:))-state(3,:)];


end

