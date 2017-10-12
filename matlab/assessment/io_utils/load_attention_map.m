function [ attention_map ] = load_attention_map(num_seq, num_frame, which_map)
%LOAD_ATTENTIONAL_MAP read an attentional map
%
%   LOAD_ATTENTIONAL_MAP Reads an attentional map given the sequence, the frame number
%   and the `which_map` parameter in [`groundtruth`, `prediction`, `central_baseline`]
    
    dreyeve_root = '/majinbu/public/DREYEVE/'; % todo avoid hardcode 
    
    seq_str = sprintf('%02d', num_seq);
        
    switch which_map
        
        case 'groundtruth'
            frame_str     = sprintf('%06d.png', num_frame + 1); % NOTICE + 1
            frame_path    = fullfile(dreyeve_root, 'DATA', seq_str, 'saliency_fix', frame_str);
            attention_map = imread(frame_path);
            
        case 'prediction'
            npz_filename  = sprintf('%06d.npz', num_frame);
            npz_path      = fullfile(dreyeve_root, 'PREDICTIONS_2017', seq_str, 'dreyeveNet', npz_filename);
            attention_map = unzip_and_load_npz(npz_path);
            
            attention_map = squeeze(attention_map);
            attention_map = attention_map ./ max(attention_map(:)); % last activation is ReLu
            attention_map = attention_map * 255;
            attention_map = uint8(attention_map);
            
        case 'central_baseline'
            frame_path    = fullfile(dreyeve_root, 'DATA', 'dreyeve_mean_train_gt_fix.png');
            attention_map = imread(frame_path);
                
        otherwise
            fprintf(2, sprintf('`which_map`=%s is not allowed.`\n', which_map))
    end
    
end

