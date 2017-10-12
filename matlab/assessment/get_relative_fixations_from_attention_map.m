function [ fixations_relative ] = get_relative_fixations_from_attention_map(attention_map)
%GET_RELATIVE_FIXATIONS_FROM_ATTENTION_MAP get coordinates of the highest
% fixations starting from the continuos attentional map
    
    map_shape = size(attention_map);
    
    attention_map_flat = attention_map(:);
    [~, fixation_idx_flat]  = sort(attention_map_flat, 'descend');
    
    top_fixation_locations = fixation_idx_flat(1:25);
    
    % Determines the equivalent subscript values corresponding to a single index into an array.
    [fix_rows, fix_cols] = ind2sub(map_shape, top_fixation_locations);
    
    % Turn into relative (range [0, 1])
    fix_rows = fix_rows ./ map_shape(1);
    fix_cols = fix_cols ./ map_shape(2);
    
    fixations_relative = [fix_rows, fix_cols];
end

