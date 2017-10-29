%   This script computes the mean reprojection error (in terms of euclidean distance)
%   introduced by the homography estimation when projecting from garmin to
%   garmin

clear; close all; clc;

% Add packages to path
addpath(genpath('homography_utils'));
addpath(genpath('vlfeat-0.9.20'));

% Parameters
dreyeve_data_root = '/majinbu/public/DREYEVE/DATA';
all_sequences = 1:74;
n_sequences = numel(all_sequences);

% For each sequence, we store mean (col 1) and variance (col 2) of error
error_means_and_vars = zeros(n_sequences, 2);
for s=1:n_sequences
    
    seq = all_sequences(s);
    
    % Root for this sequence
    seq_root = fullfile(dreyeve_data_root, sprintf('%02d', seq));
    
    % List etg and garmin sift files
    sift_gar_li = dir(fullfile(seq_root, 'sift', '*.mat'));
    n_frames = numel(sift_gar_li);
    
    % Initialize counters
    r_m = 0;  % running mean
    r_v = 0;  % running variance
    n   = 0;  % number of total matches
    
    for f = 1 :100: n_frames
        s1 = load(fullfile(seq_root, 'sift', sift_gar_li(f).name));
        fprintf(sprintf('Sequence %02d, frame %06d of %06d...\n', seq, f, n_frames));
        
        for k = min([n_frames, f+1]) : min([n_frames f+(25-1) / 2])
            
            s2 = load(fullfile(seq_root, 'sift', sift_gar_li(k).name));
            
            [matches, ~] = vl_ubcmatch(s1.sift_gar.d1,s2.sift_gar.d1);
            X1 = s1.sift_gar.f1(1:2, matches(1,:)); X1(3,:) = 1; X1([1 2], :) = X1([1 2], :)*2;
            X2 = s2.sift_gar.f1(1:2, matches(2,:)); X2(3,:) = 1; X2([1 2], :) = X2([1 2], :)*2;
            
            try
                [H, ok] = ransacfithomography(X1, X2, 0.05);
                if size(ok, 2) >= 8
                    
                    % Extract only matches that homography considers inliers
                    X1 = X1(:, ok);
                    X2 = X2(:, ok);
                    
                    % Project
                    X1_proj = H * X1;
                    X1_proj = X1_proj ./ repmat(X1_proj(3, :), 3, 1);
                    
                    % Compute error
                    errors = sqrt(sum((X1_proj - X2).^2, 1));
                    errors(isnan(errors)) = [];
                    for e=1:size(errors, 2)
                        % update mean and variance
                        n = n+1;
                        delta = errors(1, e) - r_m;
                        r_m = r_m + delta / n;
                        delta2 = errors(1, e) - r_m;
                        r_v = r_v + delta * delta2;
                    end
                end
            catch ME
                warning('Catched exception, skipping some frames');
            end
            
        end
    end
    
    % Set mean and var for this sequence
    r_v = r_v / (n-1);
    error_means_and_vars(s, 1) = r_m;
    error_means_and_vars(s, 2) = r_v;
    
end

save('error_means_and_vars_gar_to_gar', 'error_means_and_vars');

