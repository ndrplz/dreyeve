function svissetsrc_mexgen (rhs1, rhs2)
% SVISSETSRC    Set a codec's source image
%
% SVISSETSRC(C,SRC) sets codec C's source image to SRC.
%
% The image SRC must be of type UINT8.
%
% The image dimensions must match those of the image specified in
% SVISCODEC.
%
% The SVISSETSRC function is not required to encode an image.  For
% example, specifying a single, fixed image may be done with the
% SVISCODEC function alone.
%
% However, if you are encoding an image sequence, for example, you
% will likely call SVISCODEC to allocate a codec and specify only the
% first frame.  You would then call SVISSETSRC to specify subsequent
% source frames in the sequence:
%
%     svisinit
%     c=sviscodec(images(:,:,1));
%     svissetresmap(c,resmap);
%     ...
%     while not(done)
%         svissetsrc(c,images(:,:,frame));
%         d=svisencode(c,row,col);
%         ...
%     end
%     svisrelease
%
% SEE ALSO: SVISCODEC, SVISENCODE

% Mexgen generated this file on Tue Jun 10 14:13:21 2008
% DO NOT EDIT!

svismex (4, rhs1, rhs2);
