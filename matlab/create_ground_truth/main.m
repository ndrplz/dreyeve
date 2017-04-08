clear; close all; clc;

% setup vlfeat for SIFT extraction and matching
addpath('./homography');                                  
run('vlfeat-0.9.20/toolbox/vl_setup');         

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

% directories
dataset_root = 'Z:/DATA';                   % dataset (input) folder
output_root = 'X:/dreyeve_GT';              % output folder where GT will be saved
if ~exist(output_root, 'dir')
    mkdir(output_root);
end
config('dataset_root') = dataset_root;
config('output_root')  = output_root;

% load info on the maximum y that makes sense for each current run
load('highest_y_allowed.mat')

for cur_run =  num_of_runs : -1 : 1

    % parameters relative to creation of ground truth 
    config('sigma')            = 200 * px;     % spatial variance in saliency maps
    config('time_integration') =   25 * frame; % number of frames to integrate over in the heat map (must be odd)
    config('var_t')            = (config('time_integration') - 1) / 2 * frame;  % temporal diffusion decay in saliency maps
    config('highest_y')        = round(lowest_y_allowed(cur_run, 2));
    
    %create_new_ground_truth(cur_run, config);
    script_for_fixation_images(cur_run, config);
    
end
