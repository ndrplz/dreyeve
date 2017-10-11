function [ foveated_image ] = filter_multifovea(image, fix_locations)

rows = size(image, 1);
cols = size(image, 2);

svisinit

c = sviscodec(image);
           
r_multi = multi_svisresmap(rows, cols, fix_locations);

%figure; title('Current map'), imagesc(r)

svissetresmap(c, r_multi);

foveated_image = svisencode(c, rows / 2, cols / 2);

svisrelease;




