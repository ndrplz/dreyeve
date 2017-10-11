function [fn,halfres]=get_params(fn_list)
% Example programs helper function

% Copyright (C) 2006
% Center for Perceptual Systems
% University of Texas at Austin
%
% jsp Thu Sep 21 10:31:23 CDT 2006

% Prompt for source image
[index,ok]=listdlg('promptstring','Select the Input Image',...
    'selectionmode','single',...
    'listsize',[400 100],...
    'liststring',fn_list);

if not(ok)
    error('Aborted by user');
end

fn=fn_list{index};

% Prompt for halfres
[halfres,ok]=listdlg('promptstring',...
    'Select the display resolution.',...
    'selectionmode','single',...
    'listsize',[150 150],...
    'liststring',{'1','2','3','4','5','6','7','8','9','10'});

if not(ok)
    error('Aborted by user');
end
