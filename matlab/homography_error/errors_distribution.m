%   This script computes the mean reprojection error (in terms of euclidean distance)
%   introduced by the homography estimation when projecting from garmin to
%   garmin

clear; close all; clc;

% Add packages to path
addpath(genpath('homography_utils'));
addpath(genpath('vlfeat-0.9.20'));

% Parameters
dreyeve_data_root = '/majinbu/public/DREYEVE/DATA';
seq = randi([1, 74]);
n_frames = 100;

% Root for this sequence
seq_root = fullfile(dreyeve_data_root, sprintf('%02d', seq));

% List etg and garmin sift files
sift_etg_li = dir(fullfile(seq_root, 'etg', 'sift', '*.mat'));
sift_gar_li = dir(fullfile(seq_root, 'sift', '*.mat'));

% Initialize parameters
n   = 0; % running number of errors
r_m = 0; % running mean
r_v = 0; % running variance

all_errors      = []; % collection of all errors
error_to_frame  = []; % maps each error to its frame idx

% Loop over list
for f=1:n_frames
    
    fprintf(1, sprintf('Sequence %02d, frame %06d of %06d...\n', seq, f, n_frames));
    
    % Extract frame index
    f_idx = randi([0, 7499]);
    
    % Load sift files for both etg and garmin
    load(fullfile(seq_root, 'etg', 'sift', sift_etg_li(f_idx).name));
    load(fullfile(seq_root, 'sift', sift_gar_li(f_idx).name));
    
    % Compute matches
    [matches, scores] = vl_ubcmatch(sift_etg.d1,sift_gar.d1);
    
    % Prepare data in homogeneous coordinates for RANSAC
    X1 = sift_etg.f1(1:2, matches(1,:)); X1(3,:) = 1; X1([1 2], :) = X1([1 2], :)*2;
    X2 = sift_gar.f1(1:2, matches(2,:)); X2(3,:) = 1; X2([1 2], :) = X2([1 2], :)*2;
    
    try
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
            for e=1:size(errors, 2)
                % update mean and variance
                n = n+1;
                delta = errors(1, e) - r_m;
                r_m = r_m + delta / n;
                delta2 = errors(1, e) - r_m;
                r_v = r_v + delta * delta2;
            end
            
            all_errors = [all_errors; errors'];
            error_to_frame = [error_to_frame; ones(size(errors,2),1)*f_idx];
        end
    catch ME
        warning('Catched exception, skipping some frames');
    end
end

r_v = r_v / (n-1);
hist(all_errors, 100);





