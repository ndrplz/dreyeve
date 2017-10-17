%   This script computes the mean reprojection error (in terms of euclidean distance)
%   introduced by the homography estimation when projecting from garmin to
%   garmin

clear; close all; clc;

% Add packages to path
addpath(genpath('homography_utils'));
addpath(genpath('vlfeat-0.9.20'));

% Parameters
dreyeve_data_root = '/majinbu/public/DREYEVE/DATA';

% Loop over sequences
all_sequences = 1:74;
n_sequences = numel(all_sequences);
mean_sequence_error = zeros(n_sequences, 1);
for s=1:n_sequences
    
    seq = all_sequences(s);
    
    % Root for this sequence
    seq_root = fullfile(dreyeve_data_root, sprintf('%02d', seq));
    
    % List etg and garmin sift files
    sift_gar_li = dir(fullfile(seq_root, 'sift', '*.mat'));
    n_frames = numel(sift_gar_li);
    
    % rows are frames, cols are (mean error, number of matches)
    sequence_frame_sum_error = zeros(n_frames, 2);
    
    for f = 1 :100: n_frames
        s1 = load(fullfile(seq_root, 'sift', sift_gar_li(f).name));
        fprintf(sprintf('Sequence %02d, frame %06d of %06d...\n', seq, f, n_frames));
        
        for k = min([n_frames, f+1]) : min([n_frames f+(25-1) / 2])
            
            s2 = load(fullfile(seq_root, 'sift', sift_gar_li(k).name));
            
            [matches, ~] = vl_ubcmatch(s1.sift_gar.d1,s2.sift_gar.d1);
            X1 = s1.sift_gar.f1(1:2, matches(1,:)); X1(3,:) = 1; X1([1 2], :) = X1([1 2], :)*2;
            X2 = s2.sift_gar.f1(1:2, matches(2,:)); X2(3,:) = 1; X2([1 2], :) = X2([1 2], :)*2;
            
            if size(matches, 2) >= 4
                [H, ok] = ransacfithomography(X1, X2, 0.05);
                if size(ok, 2) < 8, H = zeros(3); end % sanity check
                
                
                if 0
                    % Check that the homography computed now is the same as the one
                    % stored on file. Assertion: the sum of absolute difference is
                    % lesser 1e-7.
                    stored_H_filename = fullfile(seq_root, 'homography', sprintf('gar_%06d_gar_%06d.mat', f, k));
                    stored_H = load(stored_H_filename);
                    stored_H = stored_H.H_struct.H;
                    sad = sum(sum(abs(H - stored_H)));
                    assert(sad < 1e-7, sprintf('%s is different from the one computed now!', stored_H_filename));
                end
                
                % Extract only matches that homography considers inliers
                X1 = X1(:, ok);
                X2 = X2(:, ok);
                
                % Project
                X1_proj = H * X1;
                X1_proj = X1_proj ./ repmat(X1_proj(3, :), 3, 1);
                
                % Compute projection error
                error = sqrt(sum((X1_proj - X2).^2, 1));
                sequence_frame_sum_error(k, 1) = sequence_frame_sum_error(k, 1) + nansum(error);
                sequence_frame_sum_error(k, 2) = sequence_frame_sum_error(k, 2) + size(error, 2);
            end
        end
    end
    
    % Average across frame errors
    this_sequence_mean_error = sum(sequence_frame_sum_error(:, 1)) / sum(sequence_frame_sum_error(:, 2));
    
    % Set into mean sequence error
    mean_sequence_error(s) = this_sequence_mean_error;
    
end

save('average_error_per_sequence_gar_to_gar', 'mean_sequence_error');

