function [ garmin_resmap, etg_resmap] = create_warped_resmap(seq, frames_etg, frames_garmin, fixations_relative )
%CREATE_WARPED_RESMAP Creates a resmap for the frames in etg and warps them
%according to the estimated homography

config = load_config();
etg_shape = config.etg_shape;
garmin_shape = config.garmin_shape;
dreyeve_data_root = config.dreyeve_data_root;

homography_root = fullfile(dreyeve_data_root, sprintf('%02d', seq), 'homography');

garmin_resmap = uint8(zeros(garmin_shape(1), garmin_shape(2)));
etg_resmap = uint8(zeros(etg_shape(1), etg_shape(2)));

for i=1:numel(frames_etg) % for each etg frame
    
    cur_etg_frame       = frames_etg(i);
    cur_garmin_frame    = frames_garmin(i);
    
    % create a resmap relative to etg
    cur_etg_resmap = svisresmap_multifovea(etg_shape(1), etg_shape(2), fixations_relative(i, :), 120);
    
    % load homography matrix from cur_etg_frame to cur_garmin_frame
    homography_file = fullfile(homography_root, sprintf('gar_%06d_etg_%06d.mat', cur_garmin_frame, cur_etg_frame));
    if exist(homography_file, 'file') == 2
        H = load(homography_file);
        H = H.H_struct.H;  % nice!
        
        if det(H') ~= 0
            % warp the resmap
            Rout = imref2d(garmin_shape);
            cur_garmin_resmap = imwarp(cur_etg_resmap, projective2d(H'), 'outputView',Rout);
            
            garmin_resmap   = max(garmin_resmap, cur_garmin_resmap);
            etg_resmap      = max(etg_resmap, cur_etg_resmap);
        else
            warn_msg = sprintf('Homography matrix is singular: %s', homography_file);
            warning(warn_msg);
        end
    
    end
end


end

