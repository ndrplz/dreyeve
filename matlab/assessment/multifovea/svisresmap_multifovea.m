function r_multi = svisresmap_multifovea(rows, cols, fix_locations)
% MULTI_SVISRESMAP creates a resolution map with multiple foveal locations.
%   r_multi = MULTI_SVISRESMAP(rows, cols, fix_locations)

% Here `rows` and `cols` are rows and cols of the original image

% "Typically, the ROWS and COLS parameters will be double the pixel
% resolution of the image being processed in order to accomodate
% fixations across the entire image" (from `svisresmap` docs).

% in this case 'r_base' has size of the original image
r_base  = svisresmap(rows, cols, 'halfres', 2.3, 'maparc', 230.8);

num_fixations = size(fix_locations, 1);

% Thus `r_multi` must be the double of `r_base` to allow all possible 
% translation from (0, 0) to (rows, cols). 
% There is one channel for each fixation, s.t. maximum can be computed on
% channel axis.
r_multi = zeros(rows * 2, cols * 2, num_fixations);

for i = 1 : num_fixations
    
    fix_row = fix_locations(i, 1);
    fix_col = fix_locations(i, 2);
    
    start_row = fix_row * rows - rows / 2 + rows / 2 + 1;
    start_col = fix_col * cols - cols / 2 + cols / 2 + 1;
    
    % cast to int to avoid warning
    start_row   = cast(start_row, 'int32');
    end_row   = start_row + rows - 1;
    start_col   = cast(start_col, 'int32');
    end_col   = start_col + cols - 1;
    
    try
        r_multi(start_row:end_row, start_col:end_col, i) = r_base;
    catch
        warn_msg = 'Inconsistent indexing when creating resmap. Discarding it.';
        warning(warn_msg);
    end
end

% Compute the max value across channels
r_multi = max(r_multi, [], 3);

% Crop to restore original size
r_multi = r_multi(rows/2 : rows + rows/2 - 1, cols/2 : cols + cols/2 - 1);

% Be sure to cast to uin8 before returning
r_multi = uint8(r_multi);
