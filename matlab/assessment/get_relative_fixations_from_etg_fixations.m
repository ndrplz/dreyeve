function [ etg_frames, garmin_frames, fixations_relative ] = get_relative_fixations_from_etg_fixations(seq_fixation_data, num_frame)
%GET_RELATIVE_FIXATIONS_FROM_ETG_FIXATIONS get coordinates of fixations on 
% ETG for the frame `num_frame`.

etg_shape = [720, 960];

% load all (x, y) associated to `num_frame`
data = cell2mat(seq_fixation_data(cat(1, seq_fixation_data{:, 2}) == num_frame, 1:4));
data = data(~any(isnan(data), 2), :); % remove rows that contain nans

if ~isempty(data)
    
    etg_frames      = data(:, 1);
    garmin_frames   = data(:, 2);
    fix_rows        = data(:, 3);
    fix_cols        = data(:, 4);
    
    % Turn into relative (range [0, 1])
    fix_rows = fix_rows ./ etg_shape(1);
    fix_cols = fix_cols ./ etg_shape(2);
    
    fixations_relative = [fix_rows, fix_cols];
else
    etg_frames          = [];
    garmin_frames       = [];
    fixations_relative  = [];
end
end
