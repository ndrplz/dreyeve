function hfig=setup_figure(rows,cols)
% Example programs helper function
%
% Copyright (C) 2006
% Center for Perceptual Systems
% University of Texas at Austin
%
% jsp Thu Sep 21 10:39:19 CDT 2006

% Show the dialog
figure('menubar','none')

% Set some properties
set(gcf,'Pointer','crosshair');
set(gcf,'WindowStyle','modal')
set(gcf,'WindowButtonMotionFcn',@window_button_motion_callback);
set(gcf,'KeyPressFcn',@key_press_callback);
set(gcf,'DeleteFcn',@delete_callback);

% Make the image take up the entire figure window
set(gca,'Position',[0 0 1 1])

% Center it and make the pixels in the image match the pixels on
% the screen (approximately)
screen_size=get(0,'ScreenSize');
set(gcf,'Position',[screen_size(3)/2-cols/2 screen_size(4)/2-rows/2 cols rows]);

% Return a handle to the image
hfig=image(zeros(rows,cols));
