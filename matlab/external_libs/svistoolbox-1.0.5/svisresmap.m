function m=svisresmap(rows,cols,varargin)
% SVISRESMAP    Create a resolution map
%
% R=SVISRESMAP(ROWS,COLS) will create a space variant resolution map
% that has ROWS pixel rows and COLS pixel columns.
%
% Typically, the ROWS and COLS parameters will be double the pixel
% resolution of the image being processed in order to accomodate
% fixations across the entire image.
%
% Optional parameters may be specified with 'string'/value pairs.  For
% example,
%
%   >> svisresmap(1280,960,'maparc',40);
%
% The optional parameters are:
%
% 'maparc'  Horizontal visual angle represented by the map.  Default
%           is 60 degrees.  The map usually covers twice the visual
%           angle of the image to which it is applied.  For example,
%           if the image covers 20 degrees of visual angle, the map
%           would need to be about 40 degrees.
%
%           Note also that the viewing distance of the observer may be
%           calculated from the maparc and rows parameters.
%
% 'halfres' Half resolution of the map.  This parameter specifies the
%           eccentricity at which resolution drops to half the
%           resolution in the center of the fovea.  For humans, the
%           half resolution is 2.3 degrees.  2.3 is also the default
%           value for this parameter.
%
% SEE ALSO: SVISSETRESMAP

% Copyright (C) 2004-2006 Center for Perceptual Systems
%
% jsp Wed Apr  7 15:23:20 CDT 2004

% Set defaults
maparc=60;
halfres=2.3;

% Get optional arguments
for argnum=1:2:nargin-3
    arg=varargin{argnum};
    switch lower(arg)
        case{'maparc','arc'}
        maparc=varargin{argnum+1};
        case{'halfres','halfresolution'}
        halfres=varargin{argnum+1};
        otherwise
        error(['An invalid option was specified: ' varargin{argnum}]);
    end
end

% Initialize the map
m = zeros(rows, cols);

% Compute the visual angle of a foveal pixel
pixarc = maparc / cols;

% Convert the half resolution to pixels
halfres = halfres / pixarc;

% Make matrices of x and y coordinates
[x,y] = meshgrid(1-cols/2:cols/2, 1-rows/2:rows/2);

% Compute eccentricities at each point
ecc = sqrt(x.^2 + y.^2);

m = (halfres ./ (halfres + ecc));

% Turn into unsigned chars
m = uint8(m*255);
