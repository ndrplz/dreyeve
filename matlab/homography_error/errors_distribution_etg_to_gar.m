%   This script computes the mean reprojection error (in terms of euclidean distance)
%   introduced by the homography estimation when projecting from garmin to
%   garmin

clear; close all; clc;

% Add packages to path
addpath(genpath('homography_utils'));
addpath(genpath('vlfeat-0.9.20'));

% Parameters
dreyeve_data_root = '/majinbu/public/DREYEVE/DATA';
n_frames = 100; % number of sampled frames for each sequence

all_errors      = []; % collection of all errors
errors_seq      = []; % for each error, its sequence
errors_frame    = []; % for each error, its frame

% Loop over sequences
for seq=1:74
    
    % Root for this sequence
    seq_root = fullfile(dreyeve_data_root, sprintf('%02d', seq));
    
    % List etg and garmin sift files
    sift_etg_li = dir(fullfile(seq_root, 'etg', 'sift', '*.mat'));
    sift_gar_li = dir(fullfile(seq_root, 'sift', '*.mat'));
    
    % Loop over list
    for f=1:n_frames
        
        fprintf(1, sprintf('Sequence %02d, frame %06d of %06d...\n', seq, f, n_frames));
        
        % Extract frame index
        f_idx = randi([0, 7499]);
        
        try
            % Load sift files for both etg and garmin
            load(fullfile(seq_root, 'etg', 'sift', sift_etg_li(f_idx).name));
            load(fullfile(seq_root, 'sift', sift_gar_li(f_idx).name));
            
            % Compute matches
            [matches, scores] = vl_ubcmatch(sift_etg.d1,sift_gar.d1);
            
            % Prepare data in homogeneous coordinates for RANSAC
            X1 = sift_etg.f1(1:2, matches(1,:)); X1(3,:) = 1; X1([1 2], :) = X1([1 2], :)*2;
            X2 = sift_gar.f1(1:2, matches(2,:)); X2(3,:) = 1; X2([1 2], :) = X2([1 2], :)*2;
            
            
            % Fit ransac and find homography
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
                
                all_errors = [all_errors; errors'];
                errors_seq  = [errors_seq; ones(size(errors,2), 1)* seq];
                errors_frame  = [errors_frame; ones(size(errors,2), 1)* f_idx];
            end
        catch ME
            warning('Catched exception, skipping some frames');
        end
    end
end

%% Plot the distribution as it is
histo = histogram(all_errors, 100, 'Normalization', 'pdf');
xlabel('Projection Error (ED)')
ylabel('Probability')
title('Distribution of errors etg->gar')
saveas(gcf, 'error_distribution_etg_to_gar')

% find the mode
[~, mode_idx] = max(histo.Values);
distr_mode = [histo.BinEdges(mode_idx), histo.BinEdges(mode_idx+1)]

%% Plot the distribution nicely
all_errors_nc = all_errors(all_errors < 100);
histogram(all_errors_nc, 100, 'Normalization', 'pdf');
xlabel('Projection Error (ED)')
ylabel('Probability')
title('Distribution of errors etg->gar')
saveas(gcf, 'error_distribution_etg_to_gar_nc')

%% Sort by errors and save on file problematic frames
[all_errors_sorted, sorted_idx] = sort(all_errors, 'descend');
errors_seq_sorted = errors_seq(sorted_idx);
errors_frame_sorted = errors_frame(sorted_idx);
to_save = cat(2, all_errors_sorted(1:100), errors_seq_sorted(1:100), errors_frame_sorted(1:100));
save('high_errors_etg_to_gar', 'to_save');

%% Compute quantiles
distr_quantile_50 = quantile(all_errors, 0.5)
distr_quantile_75 = quantile(all_errors, 0.75)
distr_quantile_80 = quantile(all_errors, 0.8)
distr_quantile_90 = quantile(all_errors, 0.9)
distr_quantile_95 = quantile(all_errors, 0.95)



