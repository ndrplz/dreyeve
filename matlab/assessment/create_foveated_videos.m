clear; close all; clc;

% Add Space Variant Imaging toolbox to the matlab path
addpath(genpath('../external_libs/svistoolbox-1.0.5'))

% Add custom functions to apply multifovea
addpath(genpath('./multifovea'))

% Add utils functions for data i/o
addpath(genpath('./io_utils'))

% Import python modules needed
cd('./python_interop')
utils = py.importlib.import_module('assessment_utils');
py.reload(utils); % useful if changed
cd('..')

% Output parameters
output_root    = '/majinbu/public/DREYEVE/QUALITY_ASSESSMENT_VIDEOS_MATLAB';
output_logfile = fullfile(output_root, 'videos.txt');
mkdir(output_root);
n_videos = 200;

% Load assesment experiment configuration
config = load_config();

for v = 1 : n_videos
    
    fprintf(1, sprintf('Creating video %02d...\n', v));

    % Sample a sequence and a start frame
    python_tuple = utils.get_random_clip();
    seq = double(python_tuple{1});
    start_frame = double(python_tuple{2});
    is_acting = char(python_tuple{3});
    count_acting = double(python_tuple{4});

    % Get driver for sampled sequence
    python_str = utils.get_driver_for_sequence(seq);
    driver_id  = char(python_str);

    % Sample an attentional map
    maps = {'groundtruth', 'prediction', 'central_baseline'};
    which_map = maps{randi(numel(maps))};

    % Length of the sequence
    n_frames  = double(utils.n_frames);
    end_frame = start_frame + n_frames; 
        
    % Create video filename
    seq_str         = sprintf('%02d', seq);
    start_frame_str = sprintf('%06d', start_frame);
    end_frame_str   = sprintf('%06d', end_frame);
    video_filename = concat_video_signature({driver_id, which_map, seq_str, start_frame_str, end_frame_str, is_acting}, '_') ;
    video_filename = strcat(video_filename, '.avi');

    % Open video reader
    output_video = VideoWriter(fullfile(output_root, video_filename));
    output_video.FrameRate = 25;
    open(output_video)
    
    % Average resolution map area for current sequence
    sequence_area = 0;
    
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
        [foveated_frame, sum_resmap] = filter_multifovea_rgb(dreyeve_frame, fixations_relative);
        sequence_area = sequence_area + sum_resmap;
        
        % Show result for debug
        if config.verbose
            subplot(311), imshow(dreyeve_frame), subplot(312), imshow(attention_map), subplot(313), imshow(foveated_frame)
            drawnow
        end

        % Write foveated image on video
        writeVideo(output_video, foveated_frame);

    end
    close(output_video);
    
    % Compute the average area of resolution map in the past sequence
    sequence_area = sequence_area / n_frames;

    % Create video signature
    video_signature = concat_video_signature({video_filename, driver_id, which_map, seq_str, start_frame_str, end_frame_str, is_acting, sequence_area, sprintf('%06d', count_acting)}, ';') ;
    
    % Log current video signature
    save_video_line_on_log_file(output_logfile, video_signature);
end
