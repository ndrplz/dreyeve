close all; clear; clc;

% Add packages to path
addpath(genpath('homography_utils'));
addpath(genpath('vlfeat-0.9.20'));

% Parameters
dreyeve_data_root = '/majinbu/public/DREYEVE/DATA';

% Load high_errors
high_errors = load('high_errors_etg_to_gar.mat');
high_errors = high_errors.to_save;

% Loop over errors
for e=1:size(high_errors,1)
    error = high_errors(e, 1);
    seq   = high_errors(e, 2);
    frame = high_errors(e, 3);
    
    % Root for this sequence
    seq_root = fullfile(dreyeve_data_root, sprintf('%02d', seq));
    
    % List etg and garmin sift files
    sift_etg_li = dir(fullfile(seq_root, 'etg', 'sift', '*.mat'));
    sift_gar_li = dir(fullfile(seq_root, 'sift', '*.mat'));
    
    % Load sift files for both etg and garmin
    load(fullfile(seq_root, 'etg', 'sift', sift_etg_li(frame).name));
    load(fullfile(seq_root, 'sift', sift_gar_li(frame).name));
    
    % Compute matches
    [matches, scores] = vl_ubcmatch(sift_etg.d1,sift_gar.d1);
    
    % Prepare data in homogeneous coordinates for RANSAC
    X1 = sift_etg.f1(1:2, matches(1,:)); X1(3,:) = 1; X1([1 2], :) = X1([1 2], :)*2;
    X2 = sift_gar.f1(1:2, matches(2,:)); X2(3,:) = 1; X2([1 2], :) = X2([1 2], :)*2;
    
    % Fit ransac and find homography
    [H, ok] = ransacfithomography(X1, X2, 0.05);
    X1 = X1(:, ok);
    X2 = X2(:, ok);
    
    %% Visualize stuff
    
    % Load images
    i1_path = fullfile(seq_root, 'etg', 'frames', sprintf('%06d.jpg', sift_etg.frame));
    i1 = imresize(imread(i1_path), 0.5);
    i2_path = fullfile(seq_root, 'frames', sprintf('%06d.jpg', sift_gar.frame));
    i2 = imresize(imread(i2_path), 0.5);
    
    i1_container = zeros(size(i2));
    i1_container(1:360, 1:480, :) = i1;
    
    figure(1) ; clf ;
    imagesc(cat(2, i1_container, i2)) ;
    
    xa = sift_etg.f1(1,matches(1,:)) ;
    xb = sift_gar.f1(1,matches(2,:)) + size(i2,2) ;
    ya = sift_etg.f1(2,matches(1,:)) ;
    yb = sift_gar.f1(2,matches(2,:)) ;
    
    hold on ;
    h = line([xa ; xb], [ya ; yb]) ;
    set(h,'linewidth', 1, 'color', 'b') ;
    
    vl_plotframe(sift_etg.f1(:,matches(1,:))) ;
    sift_gar.f1(1,:) = sift_gar.f1(1,:) + size(i2,2) ;
    vl_plotframe(sift_gar.f1(:,matches(2,:))) ;
    axis image off ;
    
    pause;
    
    
end