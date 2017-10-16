function [ foveated_image ] = filter_multifovea(image, fix_locations, maparc)
% [ foveated_image ] = FILTER_MULTIFOVEA(image, fix_locations) apply
% foveating filter to input image and returns the foveated image. 
%
% In case size(fix_locations, 1) > 1, multiple foveal regions are applied
% in the indicated fixation locations

rows = size(image, 1);
cols = size(image, 2);

% Init toolbox dependencies
svisinit

% Create a space variant imaging system codec
c = sviscodec(image);
           
% Create a multi-fovea resolution map according to given fixations
r_multi = svisresmap_multifovea(rows, cols, fix_locations, maparc);

% Set resolution map
svissetresmap(c, r_multi);

% Encode, that is foveated image is created here
foveated_image = svisencode(c, rows / 2, cols / 2);

% Release toolbox dependencies
svisrelease;
