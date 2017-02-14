clear; close all; clc;

% loop over each run of the dataset and annotate the lower row (higher y)
% in which is possible to have GT

% when computing saliency maps, every y coordinate that is greater than the
% highest y allowed will be treated as projection error and discarded 

dataset_root = 'Z:/DATA';                   % dataset (input) folder
num_runs = 74;

figure(1)

highest_y_allowed = zeros(num_runs, 2);

for cur_run = 1 : num_runs
    
    mean_frame_file = fullfile(dataset_root, sprintf('%02d', cur_run), sprintf('%02d_mean_frame.png', cur_run));
    
    imshow(mean_frame_file)
    drawnow
    
    [~, y] = ginput(1);
    
	highest_y_allowed(cur_run, :) = [cur_run, y];
    
end

save('highest_y_allowed.mat', 'highest_y_allowed')