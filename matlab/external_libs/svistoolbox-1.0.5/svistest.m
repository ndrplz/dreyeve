function svistest

clear; close all; clc;

% SVISTEST   Test space variant imaging system functions

% Copyright (C) 2004-2006
% Center for Perceptual Systems
%
% jsp Thu Apr  8 15:59:22 CDT 2004

img=imread('src.pgm');
ROWS=size(img,1);
COLS=size(img,2);

disp('Testing svisinit...');
svisinit;
disp('Testing svisrelease...');
svisrelease;

disp('Testing sviscodec...');

% not init'ed
fail_function('sviscodec(uint8(rand(100,100)));')

svisinit
%svisinit(1) % Pass '1' for debug info

% src is not 2D
fail_function('sviscodec(uint8(rand(4,4,4)));')

c=sviscodec(img);
disp('Testing svisresmap...');

r=svisresmap(ROWS*2,COLS*2);
r2=svisresmap(ROWS*2,COLS*2,'halfres',10.0,'maparc',80.0);
r2=uint8(255-r2);

disp('Testing svissetresmap...');

% resmap is not 2d
fail_function(sprintf('svissetresmap(%d,uint8(rand(4,4,4)));',c))
% bad handle
fail_function(sprintf('svissetresmap(%d,uint8(rand(4,4)));',c+1))

svissetresmap(c,r);

disp('Testing svisencode...');

% bad handle
fail_function(sprintf('svisencode(%d,0,0);',c+1))

dest=svisencode(c,ROWS/2,COLS/2);
h1=show(dest);

disp('Testing svissetsrc...');

% wrong size
fail_function(sprintf('svissetsrc(%d,uint8(rand(10,10)));',c))
% bad handle
fail_function(sprintf('svissetsrc(%d,uint8(rand(10,10)));',c+1))

svissetsrc(c,fliplr(img));

dest=svisencode(c,ROWS,COLS);
h2=show(dest);

% alloc another codec
c2=sviscodec(img);
%r2=uint8(rand(size(r))*255);
svissetresmap(c2,r2);
dest=svisencode(c2,ROWS/2,COLS/2);
h3=show(dest);
h4=show(img);

svisrelease;

close(h1);
close(h2);
close(h3);
close(h4);

disp 'Success';

function fail_function(s)
% Make a function call that should fail

caught=false;
try
    eval(s)
catch
    %fprintf([lasterr '\n']);
    caught=true;
end

if not(caught)
    error(['function call ''' s ''' did not fail, but should have'])
end


function h=show(i)

h=figure;
imagesc(i);
colormap(gray);
drawnow

pause
