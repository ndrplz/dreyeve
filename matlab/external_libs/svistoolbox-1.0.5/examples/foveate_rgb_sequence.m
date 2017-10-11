function foveate_rgb_sequence
% FOVEATE_RGB_SEQUENCE Realtime foveation of an image sequence
%
% Press ESC to end the demo and close the window.

% Copyright (C) 2004-2006
% Center for Perceptual Systems
% University of Texas at Austin
%
% jsp Tue Sep 19 15:00:24 CDT 2006

fprintf(2, 'TGZ files are not versioned due to their size.\n');
fprintf(2, 'To run this example consider downloading them from here: ');
fprintf(2, 'http://svi.cps.utexas.edu/software.shtml\n');
return

% Prompt for parameters
fn_list={'avner.tgz','caesar.tgz','hana.tgz','katharine.tgz'};
[fn,halfres]=get_params(fn_list);

% Uncompress the files
dn=[fn '.images'];
fprintf('Extracting %s to %s...\n',fn,dn);
untar(fn,[fn '.images']);

% Get the files into a big memory buffer
fprintf('Reading images in %s...\n',dn);
fns=dir(dn);
images=[];
for i=1:length(fns)
    if not(fns(i).isdir)
        img=imread(fullfile(dn,fns(i).name));
        if isempty(images)
            images=img;
        else
            images=cat(4,images,img);
        end
    end
end

rows=size(images,1);
cols=size(images,2);
total_frames=size(images,4);

% Initialize the library
svisinit

% Create a resmap
fprintf('Creating resolution map...\n');
resmap=svisresmap(rows*2,cols*2,'halfres',halfres);

% Break into separate color planes
red=squeeze(images(:,:,1,1));
green=squeeze(images(:,:,2,1));
blue=squeeze(images(:,:,3,1));

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
current_frame=1;
inc=1;

% Start the encoding loop
fprintf('Processing a %d X %d pixel, %d frame image sequence...\n',cols,rows,total_frames);
fprintf('Press ESC to exit...\n');

while window_open

    % Get the cursor
    pos=get(gca,'currentpoint');
    row=pos(3);
    col=pos(1);

    % Set the source
    svissetsrc(c1,images(:,:,1,current_frame));
    svissetsrc(c2,images(:,:,2,current_frame));
    svissetsrc(c3,images(:,:,3,current_frame));

    % Keep track of which frame to show next
    current_frame=current_frame+inc;
    if current_frame>total_frames
        current_frame=total_frames-1;
        inc=-1;
    elseif current_frame<1
        current_frame=2;
        inc=1;
    end
    
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

    % Slow down the frame rate
    %while frames/toc>24
    %end

    % Show frame rate
    if toc>5
        fprintf('%.1f Hz\n',frames/toc);
        frames=0;
        tic;
    end
end

% Free resources
svisrelease
