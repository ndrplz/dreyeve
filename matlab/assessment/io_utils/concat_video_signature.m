function [ video_signature ] = concat_video_signature(strings_cellarray, delimiter)
%CONCAT_VIDEO_SIGNATURE Create string of video signature
%   CONCAT_VIDEO_SIGNATURE Create string of video signature starting 
%   from randomly sampled experiment parameters
    
    video_signature = strings_cellarray{1};
    
    for i = 2 : numel(strings_cellarray)
        cell_content = strings_cellarray{i};
        cell_content_signature = sprintf('%s%s', delimiter, cell_content);
        video_signature = strcat(video_signature, cell_content_signature);
    end
    
end

