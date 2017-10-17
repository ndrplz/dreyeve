%   This script computes the mean reprojection error (in terms of euclidean distance)
%   introduced by the homography estimation when projecting from etg to
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
    sift_etg_li = dir(fullfile(seq_root, 'etg', 'sift', '*.mat'));
    sift_gar_li = dir(fullfile(seq_root, 'sift', '*.mat'));
    n_frames = numel(sift_etg_li);
    
    % rows are frames, cols are (mean error, number of matches)
    sequence_frame_sum_error = zeros(n_frames, 2);
    
    % Loop over list
    for f=1:100:n_frames
        
        fprintf(1, sprintf('Sequence %02d, frame %06d of %06d...\n', seq, f, n_frames));
        
        % Load sift files for both etg and garmin
        load(fullfile(seq_root, 'etg', 'sift', sift_etg_li(f).name));
        load(fullfile(seq_root, 'sift', sift_gar_li(f).name));
        
        % Compute matches
        [matches, scores] = vl_ubcmatch(sift_etg.d1,sift_gar.d1);
        
        %numMatches = size(matches,2) ;
        
        % Prepare data in homogeneous coordinates for RANSAC
        X1 = sift_etg.f1(1:2, matches(1,:)); X1(3,:) = 1; X1([1 2], :) = X1([1 2], :)*2;
        X2 = sift_gar.f1(1:2, matches(2,:)); X2(3,:) = 1; X2([1 2], :) = X2([1 2], :)*2;
        
        % Fit ransac and find homography
        [H, ok] = ransacfithomography(X1, X2, 0.05);
        if size(ok, 2) < 8, H = zeros(3); end % sanity check
        
        if 0
            % Check that the homography computed now is the same as the one
            % stored on file. Assertion: the sum of absolute difference is
            % lesser 1e-7.
            stored_H_filename = fullfile(seq_root, 'homography', sprintf('gar_%06d_etg_%06d.mat', sift_gar.frame, sift_etg.frame));
            stored_H = load(stored_H_filename);
            stored_H = stored_H.H_struct.H;
            sad = sum(sum(abs(H - stored_H)));
            assert(sad < 1e-7, sprintf('%s is different from the one computed now!', stored_H_filename));
        end
        
        % Extract only matches that homography considers inliers
        X1 = X1(:, ok);
        X2 = X2(:, ok);
        
        X1_proj = H * X1;
        X1_proj = X1_proj ./ repmat(X1_proj(3, :), 3, 1);
        
        
        error = sqrt(sum((X1_proj - X2).^2, 1));
        sequence_frame_sum_error(f, 1) = nansum(error);
        sequence_frame_sum_error(f, 2) = size(error, 2);

    end
    
    % Average across frame errors
    this_sequence_mean_error = sum(sequence_frame_sum_error(:, 1)) / sum(sequence_frame_sum_error(:, 2));
    
    % Set into mean sequence error
    mean_sequence_error(s) = this_sequence_mean_error; 
end

save('average_error_per_sequence_etg_to_gar', mean_sequence_error);
