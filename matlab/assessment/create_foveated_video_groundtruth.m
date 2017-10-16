function create_foveated_video_groundtruth(video_abs_path, seq, start_frame, n_frames, output_logfile, video_signature)
%CREATE_FOVEATED_VIDEO_GROUNDTRUTH Creates a foveated video for human
% fixations.
%   The gaze captured by ETG is used to compute the resmap over the ETG
%   frame. The resmap is then warped according to the estimated homography
%   to the Garmin frame and used to blur it.

config = load_config();
dreyeve_data_root = config.dreyeve_data_root;

% Open video reader
output_video = VideoWriter(video_abs_path);
output_video.FrameRate = 25;
open(output_video)

% Load gaze data for the sequence

gaze_file = fullfile(dreyeve_data_root, sprintf('%02d', seq), 'etg', sprintf('%02d_samples_cropped.txt', seq));
gaze_data = table2cell(readtable(gaze_file, 'delimiter', ' '));

if config.verbose, figure(1), end
for idx_to_load = start_frame : start_frame + n_frames
    
    % Get relative fixations (range [0, 1]) from attention maps
    % TODO check the idx_to_load (garmin) is ok to use in the gaze_data
    % (etg)
    [ etg_frames, garmin_frames, fixations_relative ] = get_relative_fixations_from_etg_fixations(gaze_data, idx_to_load);
    
    if ~isempty(fixations_relative)
        % Create and warp the resmap
        [garmin_resmap, etg_resmap] = create_warped_resmap(seq, etg_frames, garmin_frames, fixations_relative);
        
        % Load frame
        etg_frame = load_dreyeve_frame(seq, etg_frames(1), 1);
        garmin_frame = load_dreyeve_frame(seq, idx_to_load, 0);
        
        % Foveate garmin frame
        garmin_frame_foveated = filter_given_resmap_rgb(garmin_frame, garmin_resmap);
        
        % Plot
        if config.verbose
            subplot(311), imshow(im2double(0.5*etg_frame) + 0.5*ind2rgb(etg_resmap, jet(255)));
            subplot(312), imshow(im2double(0.5*garmin_frame) + 0.5*ind2rgb(garmin_resmap, jet(255)));
            subplot(313), imshow(garmin_frame_foveated);
            drawnow
        end
        
        % Write foveated image on video
        writeVideo(output_video, garmin_frame_foveated);
        
    else
        warn_msg = sprintf('No fixation points are available for seq %02d, frame %06d', seq, idx_to_load);
        warning(warn_msg);
    end
    
end
close(output_video);

save_video_line_on_log_file(output_logfile, video_signature);

end

