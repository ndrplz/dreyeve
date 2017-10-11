function foveate_rgb_image
% FOVEATE_RGB_IMAGE   Realtime foveation of a static, color image
%
% Press ESC to end the demo and close the window.

% Copyright (C) 2004-2006
% Center for Perceptual Systems
% University of Texas at Austin
%
% jsp Fri Apr 16 19:13:14 CDT 2004
% jsp Thu Sep 14 15:34:22 CDT 2006

% Prompt for source image
fn_list={'armstrongs.jpg','bee.jpg','hamiltoncreek.jpg','interspar.jpg','amstel.jpg'};
[fn,halfres]=get_params(fn_list);

% Read in the file
fprintf('Reading %s...\n',fn);
img=imread(fn);
rows=size(img,1);
cols=size(img,2);

% Break into separate color planes
red=squeeze(img(:,:,1));
green=squeeze(img(:,:,2));
blue=squeeze(img(:,:,3));

% Initialize the library
svisinit

% Create a resmap
fprintf('Creating resolution map...\n');
resmap=svisresmap(rows*2,cols*2,'halfres',halfres);

% Create 3 codecs for r, g, and b
fprintf('Creating codecs...\n');
c1=sviscodec(red);
c2=sviscodec(green);
c3=sviscodec(blue);

% The masks get created when you set the map
fprintf('Creating blending masks...\n');
svissetresmap(c1,resmap)
svissetresmap(c2,resmap)
svissetresmap(c3,resmap)

% Setup the figure window
hfig=setup_figure(rows,cols);
global window_open
window_open=true;

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
    i1=svisencode(c1,row,col);
    i2=svisencode(c2,row,col);
    i3=svisencode(c3,row,col);

    % Put them back together
    rgb=cat(3,i1,i2,i3);

    % Display it
    %image(rgb)
    set(hfig,'cdata',rgb);
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
