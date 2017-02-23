clear; close all; clc;

% add optical flow mex to path
addpath(genpath('../../external_libs/OpticalFlow/'))

px              = 1;
img_h           = 1080 * px;                % garmin image height
img_w           = 1920 * px;                % garmin image width
num_of_runs     =   74;

subsample = 16;
resize_h = img_h / subsample;
resize_w = img_w / subsample;

% directories
dataset_root = 'Z:/DATA';                   % dataset (input) folder
output_root = 'X:/dreyeve_OF';              % output folder where OF will be saved
if ~exist(output_root, 'dir')
    mkdir(output_root);
end
keys = {'dataset_root', 'output_root'};
vals = {dataset_root, output_root};
config = containers.Map(keys, vals, 'UniformValues', false);

for cur_run = 1 : 74
    
    dir_frames = fullfile(config('dataset_root'), sprintf('%02d/frames/', cur_run));
    dir_out_OF = fullfile(config('output_root'), sprintf('%02d', cur_run));
    if ~exist(dir_out_OF, 'dir')
        mkdir(dir_out_OF)
    end

    frames = dir(fullfile(dir_frames, '*.jpg'));
    for cur_frame = 1 : numel(frames) - 1 % skip last frame (todo do better)
        
        fprintf(1, '[OPTICAL FLOW]   RUN: %02d - FRAME: %06d\n', cur_run, cur_frame);

        frame_1_path = fullfile(dir_frames, frames(cur_frame).name);
        frame_2_path = fullfile(dir_frames, frames(cur_frame + 1).name);
        
%         for subsample = 2 : 2 : 16
%             resize_h = img_h / subsample;
%             resize_w = img_w / subsample;
%             tic
%             flow_img = compute_optical_flow(frame_1_path, frame_2_path, resize_h, resize_w);
%             flow_img = imresize(flow_img, [1080, 1920]);
%             figure, imshow(flow_img), title(sprintf('SUBSAMPLE %d - time %.02f', subsample, toc))
%         end

        flow_img = compute_optical_flow(frame_1_path, frame_2_path, resize_h, resize_w);
        
        imwrite(flow_img, fullfile(dir_out_OF, sprintf('%06d.png', cur_frame)))
        
        if 0
            subplot(221), imshow(frame_1);
            subplot(222), imshow(frame_2);
        end
        
    end
end



