function svisbenchmark
% SVISBENCHMARK   Benchmark space variant imaging system functions

% Copyright (C) 2004-2006
% Center for Perceptual Systems
%
% jsp Wed Sep 13 11:31:25 CDT 2006

img=imread('src.pgm');
ROWS=size(img,1);
COLS=size(img,2);

fprintf('Benchmarking a %d X %d image\n',COLS,ROWS);

svisinit;
c=sviscodec(img);
r=svisresmap(ROWS*2,COLS*2);
svissetresmap(c,r);

colormap(gray(256));

tic
n=0;
while toc<2
    loc=rand(2,1);
    d=svisencode(c,loc(1)*ROWS,loc(2)*COLS);
    n=n+1;
end

fprintf('%.1fHz (no display)\n',n/toc);

% It's (sometimes) much faster to set the image cdata property than to
% call 'image(img)'
himg=image(img);

tic
n=0;
while toc<2
    loc=rand(2,1);
    img=svisencode(c,loc(1)*ROWS,loc(2)*COLS);
    set(himg,'cdata',img);
    drawnow
    n=n+1;
end

fprintf('%.1fHz (with display)\n',n/toc);
close

svisrelease;
