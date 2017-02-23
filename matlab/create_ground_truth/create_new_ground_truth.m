function [ret] = create_new_ground_truth(num_run, config)
%CREATE_NEW_GROUND_TRUTH Re-compute Dreyeve ground truth without averaging
%   Detailed explanation goes here
    
    fprintf(1, 'RE-Computing ground truth for run: %02d...\n', num_run);
    
    % define some paths for code readability
    data_path = fullfile(config('dataset_root'), sprintf('%02d', num_run));
    dir_homography = fullfile(data_path, 'homography');
    dir_out_saliency = fullfile(config('output_root'), sprintf('%02d', num_run));
    if ~exist(dir_out_saliency, 'dir')
        mkdir(dir_out_saliency)
    end
    
    % load gaze data - data have the following structure:
    %   frame_etg | frame_gar | X | Y | event_type | code
    gaze_file = fullfile(data_path, 'etg', sprintf('%02d_samples_cropped.txt', num_run));
    gaze_data = table2cell(readtable(gaze_file, 'delimiter', ' '));
    
    max_frame = config('seq_len') * config('fps_video_gar');
    img_h = config('img_h');
    img_w = config('img_w');
    max_y_allowed = config('highest_y');
    for f = 1 : max_frame
        
        fprintf(1, 'RUN: %02d - FRAME: %06d\n', num_run, f);
        
        frame_out_path = fullfile(dir_out_saliency, sprintf('%06d.png', f));
        if exist(frame_out_path, 'file')
            continue;
        end
                
        % load all (x, y) associated to a certain frame
        data    = cell2mat(gaze_data(cat(1, gaze_data{:, 2}) == f-1, [3 4]));
        data    = data(~any(isnan(data), 2), :); % remove rows that contain nans
        
        saliency_map = zeros(img_h, img_w);
        
        % project gaze data from frame f to garmin video
        if ~isempty(data)
            file    = dir(fullfile(dir_homography, sprintf('gar_%06d_etg_*.mat', f)));
            H       = load(fullfile(dir_homography, file(1).name));
            P       = [data ones(size(data, 1), 1)]; % homogeneous coords
            p       = H.H_struct.H * P';
            pt_f    = p([1 2], :) ./ repmat(p(3, :), 2, 1);
            
            % check that y found is compatible with max y for this run
            pt_f = pt_f(:, pt_f(2, :) < max_y_allowed);
            
            if ~isempty(pt_f) && any(~isnan(pt_f(:)))
                saliency_map = create_gaussian_response(pt_f', config('sigma'), [img_h, img_w]);
            end
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
            
            % check that y found is compatible with max y for this run
            pt_k = pt_k(:, pt_k(2, :) < max_y_allowed);
            if isempty(pt_k), continue; end
            
            if k > f
                pt_f_k = H_struct.H \ [pt_k' ones(size(pt_k, 2), 1)]';
            else
                pt_f_k = H_struct.H * [pt_k' ones(size(pt_k, 2), 1)]';
            end
            
            pt_f_k = pt_f_k([1 2], :) ./ repmat(pt_f_k(3, :), 2, 1);
            
            if any(isnan(pt_f_k(:))), continue; end
                        
            this_saliency_map = create_gaussian_response(pt_f_k', config('sigma'), [img_h, img_w]);
            saliency_map  = max(saliency_map, this_saliency_map);
            % saliency_map  = imfilter(saliency_map, ones(15, 15) / (15*15));
            
        end
        
        % normalize and save
        saliency_map = saliency_map ./ (eps+max(max(saliency_map)));
        saliency_map(saliency_map < 10e-5) = 0;
        
        imwrite(saliency_map, frame_out_path);
        
        if 0
            imagesc(saliency_map), title(sprintf('max f = %d - saliency map', f));
            drawnow; %pause
        end
        
    end
    
    ret = true;
end

