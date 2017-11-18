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

% Loop over sequences
for seq=1:74
    
    % Root for this sequence
    seq_root = fullfile(dreyeve_data_root, sprintf('%02d', seq));
    
    % List etg and garmin sift files
    sift_gar_li = dir(fullfile(seq_root, 'sift', '*.mat'));
    
    % Loop over list
    for f=1:n_frames
        
        fprintf(1, sprintf('Sequence %02d, frame %06d of %06d...\n', seq, f, n_frames));
        try
            % Extract frame index
            f_idx = randi([0, 7499]);
            
            % Load sift file for first garmin frame
            s1 = load(fullfile(seq_root, 'sift', sift_gar_li(f_idx).name));
            
            for k = min([7500, f_idx+1]) : min([7500 f_idx+(25-1) / 2])
                
                % Load sift file for second garmin frame
                s2 = load(fullfile(seq_root, 'sift', sift_gar_li(k).name));
                
                
                % Compute matches
                [matches, ~] = vl_ubcmatch(s1.sift_gar.d1,s2.sift_gar.d1);
                
                % Prepare data in homogeneous coordinates for RANSAC
                X1 = s1.sift_gar.f1(1:2, matches(1,:)); X1(3,:) = 1; X1([1 2], :) = X1([1 2], :)*2;
                X2 = s2.sift_gar.f1(1:2, matches(2,:)); X2(3,:) = 1; X2([1 2], :) = X2([1 2], :)*2;
                
                
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
                end
            end
        catch ME
            warning('Catched exception, skipping some frames');
            
        end
    end
end

histo = histogram(all_errors, 100, 'Normalization', 'pdf');
xlabel('Projection Error (ED)')
ylabel('Probability')
title('Distribution of errors gar->gar')
saveas(gcf, 'error_distribution_gar_to_gar')

% find the mode
[~, mode_idx] = max(histo.Values);
distr_mode = [histo.BinEdges(mode_idx), histo.BinEdges(mode_idx+1)]

%% Plot the distribution nicely
all_errors_nc = all_errors(all_errors < 100);
histogram(all_errors_nc, 100, 'Normalization', 'pdf');
xlabel('Projection Error (ED)')
ylabel('Probability')
title('Distribution of errors gar->gar')
saveas(gcf, 'error_distribution_gar_to_gar_nc')

%% Compute quantiles
distr_quantile_50 = quantile(all_errors, 0.5)
distr_quantile_75 = quantile(all_errors, 0.75)
distr_quantile_80 = quantile(all_errors, 0.8)
distr_quantile_90 = quantile(all_errors, 0.9)
distr_quantile_95 = quantile(all_errors, 0.95)







