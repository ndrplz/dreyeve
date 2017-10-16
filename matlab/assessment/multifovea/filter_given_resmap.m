function [ foveated_image ] = filter_given_resmap( image, resmap_wrt )
%FILTER_GIVEN_RESMAP Foveates an image given a resmap rather than raw
%fixations.

rows = size(image, 1);
cols = size(image, 2);

% Init toolbox dependencies
svisinit

% Create a space variant imaging system codec
c = sviscodec(image);
           
% Set resolution map
svissetresmap(c, resmap_wrt);

% Encode, that is foveated image is created here
foveated_image = svisencode(c, rows / 2, cols / 2);

% Release toolbox dependencies
svisrelease;

end

