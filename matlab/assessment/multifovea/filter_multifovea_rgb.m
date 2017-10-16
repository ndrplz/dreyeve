function [ foveated_image_rgb, sum_resmap ] = filter_multifovea_rgb(image, fix_locations)
% [ foveated_image_rgb ] = FILTER_MULTIFOVEA_RGB(image, fix_locations) apply
% foveating filter to input RGB image and returns the foveated image. 
%
% In case size(fix_locations, 1) > 1, multiple foveal regions are applied
% in the indicated fixation locations

% Break into separate color planes
red_channel   = squeeze(image(:,:,1));
green_channel = squeeze(image(:,:,2));
blue_channel  = squeeze(image(:,:,3));

% Filter each channel separately
[red_foveated, sum_resmap] = filter_multifovea(red_channel, fix_locations);
[green_foveated, ~] = filter_multifovea(green_channel, fix_locations);
[blue_foveated,  ~] = filter_multifovea(blue_channel, fix_locations);


% Put them back together
foveated_image_rgb = cat(3, red_foveated, green_foveated, blue_foveated);
