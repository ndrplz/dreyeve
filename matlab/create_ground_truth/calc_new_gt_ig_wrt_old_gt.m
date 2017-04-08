clear; close all; clc;

%% set up config object
% define units of measures for code readability
frame   = 1;
sec     = 1;
minute  = 60 * sec;
px      = 1;

% dataset constants (no need to change this section)
seq_len         =    5 * minute;            % len of each sequence
fps_video_etg   =   30 * frame / sec;    % etg video frame rate
fps_video_gar   =   25 * frame / sec;    % garmin video frame rate
img_h           = 1080 * px;                % garmin image height
img_w           = 1920 * px;                % garmin image width
num_of_runs     =   74;

% create a configuration dictionary
keys = {'seq_len', 'fps_video_etg', 'fps_video_gar', 'img_h', 'img_w', 'num_of_runs'};
vals = [seq_len  ,  fps_video_etg ,  fps_video_gar ,  img_h ,  img_w ,  num_of_runs];
config = containers.Map(keys, vals, 'UniformValues', false);
config('dataset_root') = 'Z:/DATA';
config('sigma')            = 200 * px;     % spatial variance in saliency maps
config('time_integration') =   25 * frame; % number of frames to integrate over in the heat map (must be odd)
config('var_t')            = (config('time_integration') - 1) / 2 * frame;  % temporal diffusion decay in saliency maps

load('highest_y_allowed.mat')

%%
sum_ig = 0;
n_frames = 0;
for num_run=1:74    
    fprintf(1, 'RUN: %02d\n', num_run);
    
    data_path = fullfile(config('dataset_root'), sprintf('%02d', num_run));
    dir_homography = fullfile(data_path, 'homography');

    gaze_file = fullfile(data_path, 'etg', sprintf('%02d_samples_cropped.txt', num_run));
    gaze_data = table2cell(readtable(gaze_file, 'delimiter', ' '));

    max_frame = config('seq_len') * config('fps_video_gar');
    img_h = config('img_h');
    img_w = config('img_w');
    
    for f = 1 : max_frame
        
        fixation_points = [];
        
        % load the good old saliency map
        saliency_map = imread(fullfile(data_path, 'saliency', sprintf('%06d.png', f)));
        
        % load the new fixation map
        fixation_map = imread(fullfile(data_path, 'saliency_fix', sprintf('%06d.png', f)));

                        
        % load all (x, y) associated to a certain frame
        data    = cell2mat(gaze_data(cat(1, gaze_data{:, 2}) == f-1, [3 4]));
        data    = data(~any(isnan(data), 2), :); % remove rows that contain nans

                % load all (x, y) associated to a certain frame
        data    = cell2mat(gaze_data(cat(1, gaze_data{:, 2}) == f-1, [3 4]));
        data    = data(~any(isnan(data), 2), :); % remove rows that contain nans
                
        % project gaze data from frame f to garmin video
        if ~isempty(data)
            file    = dir(fullfile(dir_homography, sprintf('gar_%06d_etg_*.mat', f)));
            H       = load(fullfile(dir_homography, file(1).name));
            P       = [data ones(size(data, 1), 1)]; % homogeneous coords
            p       = H.H_struct.H * P';
            pt_f    = p([1 2], :) ./ repmat(p(3, :), 2, 1);
                        
        end
        
        T = config('time_integration'); % number of frames to integrate over
        num_frame_start = max([1, f - (T - 1) / 2]);
        num_frame_end   = min([max_frame, f + (T - 1) / 2]);
        
        for k = num_frame_start : num_frame_end
            
            if f == k, continue; end
            
            load(fullfile(dir_homography, sprintf('gar_%06d_gar_%06d.mat', min([f, k]), max([f, k]))));
            
            % project gaze data from frame k to frame f on gar video
            data    = cell2mat(gaze_data(cat(1, gaze_data{:, 2}) == k - 1, [3 4]));
            data    = data(~any(isnan(data), 2), :); if isempty(data), continue; end
            file    = dir(fullfile(dir_homography, sprintf('gar_%06d_etg*.mat', k)));
            H       = load(fullfile(dir_homography, file(1).name));
            P       = [data ones(size(data, 1), 1)];
            p       = H.H_struct.H * P';
            pt_k    = p([1 2], :) ./ repmat(p(3, :), 2, 1);
            

            if k > f
                pt_f_k = H_struct.H \ [pt_k' ones(size(pt_k, 2), 1)]';
            else
                pt_f_k = H_struct.H * [pt_k' ones(size(pt_k, 2), 1)]';
            end
            
            pt_f_k = pt_f_k([1 2], :) ./ repmat(pt_f_k(3, :), 2, 1);
            
            if any(isnan(pt_f_k(:))), continue; end
                        
            % append fixation points to the set of fixation points for
            % frame f
            fixation_points = [fixation_points pt_f_k];
            
        end
        
        this_frame_ig = information_gain(fixation_points, fixation_map, saliency_map);
        sum_ig = sum_ig + this_frame_ig;
        n_frames = n_frames + 1;
    end
    
    fprintf(1, 'Current IG: %05f\n', sum_ig / n_frames);
end
