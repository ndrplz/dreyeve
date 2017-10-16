function create_foveated_video_prediction(video_abs_path, seq, start_frame, n_frames, which_map, output_logfile, video_signature)
%CREATE_FOVEATED_VIDEO_PREDICTIONS Creates a foveated video for network
%prediction or central baseline.
%   This is different from the case of groundtruth, since here the 
%   foveatic point is retrieved by taking the max of prediction fixation.

config = load_config();

% Open video reader
output_video = VideoWriter(video_abs_path);
output_video.FrameRate = 25;
open(output_video)

if config.verbose, figure(1), end
for idx_to_load = start_frame : start_frame + n_frames
    
    % Load frame
    dreyeve_frame = load_dreyeve_frame(seq, idx_to_load);
    dreyeve_frame = imresize(dreyeve_frame, [1080 / 2, 1920 / 2]);
    
    % Load attentional map
    attention_map = load_attention_map(seq, idx_to_load, which_map);
    attention_map = imresize(attention_map, [1080 / 2, 1920 / 2]);
    
    % Get relative fixations (range [0, 1]) from attention maps
    fixations_relative = get_relative_fixations_from_attention_map(attention_map);
    
    % Create foveated image
    foveated_frame = filter_multifovea_rgb(dreyeve_frame, fixations_relative, 230.8);
    
    % Show result for debug
    if config.verbose
        subplot(211), imshow(attention_map), subplot(212), imshow(foveated_frame)
        drawnow
    end
    
    % Write foveated image on video
    writeVideo(output_video, foveated_frame);
    
end
close(output_video);

save_video_line_on_log_file(output_logfile, video_signature);

end

