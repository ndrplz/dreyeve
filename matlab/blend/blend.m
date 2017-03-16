%%
% This matlab script is used to blend a dreyeve sequence with predictions

frames_dir = 'Z:\DATA\';
pred_dir = 'Z:\PREDICTIONS\architecture7';
out_dir = 'out';
sequences = [26, 60];

mkdir(out_dir);

for s=1:length(sequences)
    seq = sequences(s);
    
    fprintf(1, sprintf('Processing sequence %06d\n', seq));
    
    f_dir = fullfile(frames_dir,sprintf('%02d/frames', seq));
    p_dir = fullfile(pred_dir,sprintf('%02d/output', seq));
    
    out_dir_seq = fullfile(out_dir,sprintf('%02d',seq));
    mkdir(out_dir_seq);
    
    for fr=15:7499
        frame = im2double(imread(fullfile(f_dir, sprintf('%06d.jpg', fr))));
        saliency = imread(fullfile(p_dir, sprintf('%06d.png', fr+1)));
        saliency = imresize(saliency, [1080, 1920]);
        
        [X, map] = gray2ind(saliency, 16); saliency_rgb = ind2rgb(X, jet(16));
        blended = 0.5  *frame + 0.5 * saliency_rgb ;
        imshow(blended);
        imwrite(blended, fullfile(out_dir_seq, sprintf('blend_%06d.jpg', fr)));        
    end
    
end

