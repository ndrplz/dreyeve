function [ ig ] = information_gain(fixation_points, fixation_map, saliency_map)
    close all;
    
    fixation_map = double(fixation_map);
    saliency_map = double(saliency_map);
    
    % normalize and vectorize saliency maps
    fixation_map = (fixation_map - min(fixation_map(:)))/(max(fixation_map(:))-min(fixation_map(:))); 
    saliency_map = (saliency_map - min(saliency_map(:)))/(max(saliency_map(:))-min(saliency_map(:)));

    % turn into distributions
    fixation_map = fixation_map./sum(fixation_map(:));
    saliency_map = saliency_map./sum(saliency_map(:));

    fixation_points = int32(fixation_points);
    fixation_points = fixation_points([2,1],:);
    fixation_points = fixation_points(:,fixation_points(1,:)>0 & fixation_points(1,:)<=1080);
    fixation_points = fixation_points(:,fixation_points(2,:)>0 & fixation_points(2,:)<=1920);
    
    locs=sub2ind(size(saliency_map), fixation_points(1,:), fixation_points(2,:));
    
    ig = mean(log2(eps+fixation_map(locs))-log2(eps+saliency_map(locs))); 

end

