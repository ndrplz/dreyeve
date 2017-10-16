function [ foveated_image_rgb ] = filter_given_resmap_rgb( image, resmap_wrt )
%FILTER_GIVEN_RESMAP_RGB Foveates a 3-channels image given a resmap rather than raw
%fixations.

% Break into separate color planes
red_channel   = squeeze(image(:,:,1));
green_channel = squeeze(image(:,:,2));
blue_channel  = squeeze(image(:,:,3));

% Filter each channel separately
red_foveated   = filter_given_resmap(red_channel, resmap_wrt);
green_foveated = filter_given_resmap(green_channel, resmap_wrt);
blue_foveated  = filter_given_resmap(blue_channel, resmap_wrt);

% Put them back together
foveated_image_rgb = cat(3, red_foveated, green_foveated, blue_foveated);

end

