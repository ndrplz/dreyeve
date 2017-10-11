function [lhs1] = sviscodec_mexgen (rhs1)
% SVISCODEC Create a space variant imaging system codec
%
% C=SVISCODEC(SRC) creates a codec that will encode source image SRC.
%
% SRC must be of type uint8.
%
% The return value is a handle to a codec that should be used in
% subsequent calls to the SVIS toolbox.
%
% SEE ALSO: SVISRESMAP, SVISSETRESMAP, SVISSETSRC, SVISENCODE

% Mexgen generated this file on Tue Jun 10 14:13:21 2008
% DO NOT EDIT!

[lhs1] = svismex (2, rhs1);
