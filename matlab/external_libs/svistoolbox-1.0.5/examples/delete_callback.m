function delete_callback(src,eventdata)
% Example programs helper function

% Copyright (C) 2006
% Center for Perceptual Systems
% University of Texas at Austin
%
% jsp Thu Sep 21 10:27:45 CDT 2006

% Signal the encoding loop to terminate
global window_open
window_open=false;
