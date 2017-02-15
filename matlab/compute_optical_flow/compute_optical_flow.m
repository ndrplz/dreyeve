function [flow_img] = compute_optical_flow(frame1_path, frame2_path, resize_h, resize_w)
%COMPUTE_OPTICAL_FLOW Compute optical flow for a couple of frames
%   todo detailed explanation goes here
    
    if exist(frame1_path, 'file') && exist(frame2_path, 'file')
        
        frame1 = imread(frame1_path);
        frame2 = imread(frame2_path);
        
        frame1 = imresize(frame1, [resize_h, resize_w]);
        frame2 = imresize(frame2, [resize_h, resize_w]);
        
        % compute optical flow between the two frames
        [vx, vy, ~] = get_optical_flow(frame1, frame2);
        
        % convert to optical flow image
        flow(:, :, 1) = vx;
        flow(:, :, 2) = vy;
        flow_img = flowToColor(flow);

        % save output image
        % imwrite(flow_img, out_img_path);
    else
        fprintf(2, 'Input files do not exist.');
        flow_img = 0;
    end
    
end

