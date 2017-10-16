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
n_videos = 20;

for v=1:n_videos
    
    fprintf(1, sprintf('Creating video %02d\n', v));
    
    % Sample a sequence and a start frame
    python_tuple = utils.get_random_clip();
    seq = double(python_tuple{1});
    start_frame = double(python_tuple{2});
    is_acting = char(python_tuple{3});
    
    % Get driver for sampled sequence
    python_str = utils.get_driver_for_sequence(seq);
    driver_id  = char(python_str);
        
    % Length of the sequence
    n_frames  = double(utils.n_frames);
    end_frame = start_frame + n_frames;
    
    % Sample an attentional map
    maps = {'groundtruth', 'prediction', 'central_baseline'};
    which_map = maps{randi(numel(maps))};
    
    % Create video filename
    seq_str         = sprintf('%02d', seq);
    start_frame_str = sprintf('%06d', start_frame);
    end_frame_str   = sprintf('%06d', end_frame);
    video_filename = concat_video_signature({driver_id, which_map, seq_str, start_frame_str, end_frame_str, is_acting}, '_') ;
    video_filename = strcat(video_filename, '.avi');
    
    % Create video signature
    video_signature = concat_video_signature({video_filename, driver_id, which_map, seq_str, start_frame_str, end_frame_str, is_acting}, ';') ;
    
    video_abs_path = fullfile(output_root, video_filename);
    switch which_map
        case 'groundtruth'
            % create video for groundtruth
            create_foveated_video_groundtruth(video_abs_path, seq, start_frame, n_frames, output_logfile, video_signature);            
        otherwise
            
            % Create video for prediction (or baseline)
            create_foveated_video_prediction(video_abs_path, seq, start_frame, n_frames, which_map, output_logfile, video_signature);
    end
end
