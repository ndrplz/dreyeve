clear; close all; clc;

% Add Space Variant Imaging toolbox to the matlab path
addpath(genpath('../../external_libs/svistoolbox-1.0.5'))

% Add custom functions to apply multifovea
addpath(genpath('../multifovea'))

% Load test image
image = imread('bee.jpg');

% Hello world fixations
fixations = [0.1, 0.1; 
             0.5, 0.5;
             0.75, 0.25;
             0, 1.0];
         
% Apply multi-foveal filter to color image
foveated_image = filter_multifovea_rgb(image, fixations);

% Show result
imshow(foveated_image)