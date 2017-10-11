function foveate_gs_image
% FOVEATE_GS_IMAGE   Realtime foveation of a static, grayscale image
%
% Press ESC to end the demo and close the window.

% Copyright (C) 2004-2006
% Center for Perceptual Systems
% University of Texas at Austin
%
% jsp Wed Sep 20 17:24:30 CDT 2006

% Prompt for parameters
fn_list={'c17.jpg','lu.jpg','spacewalk.jpg'};
[fn,halfres]=get_params(fn_list);

% Read in the file
fprintf('Reading %s...\n',fn);
img=imread(fn);
rows=size(img,1);
cols=size(img,2);

% Initialize the library
svisinit

% Create a resmap
fprintf('Creating resolution map...\n');
resmap=svisresmap(rows*2,cols*2,'halfres',halfres);

% Create 3 codecs for r, g, and b
fprintf('Creating codec...\n');
c=sviscodec(img);

% The masks get created when you set the map
fprintf('Creating blending masks...\n');
svissetresmap(c,resmap)

% Setup the figure window
hfig=setup_figure(rows,cols);
global window_open
window_open=true;
colormap(gray(256));

% Variables for displaying frame rate
tic;
frames=0;

% Start the encoding loop
fprintf('Processing a %d X %d pixel image...\n',cols,rows);
fprintf('Press ESC to exit...\n');

while window_open

    % Get the cursor
    pos=get(gca,'currentpoint');
    row=pos(3);
    col=pos(1);

    % Encode
    img=svisencode(c,row,col);

    % Display it
    set(hfig,'cdata',img);
    set(gca,'tickdir','out');
    drawnow

    frames=frames+1;

    % Show frame rate
    if toc>5
        fprintf('%.1f Hz\n',frames/toc);
        frames=0;
        tic;
    end
end

% Free resources
svisrelease
