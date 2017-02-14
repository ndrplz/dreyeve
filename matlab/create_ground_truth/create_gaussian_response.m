function F = create_gaussian_response(mu, var_s, resize_dim)

h = resize_dim(1);
w = resize_dim(2);

k = 5;
Sigma = [var_s / k, 0; 0, var_s / k];

max_y = round(h / k);
max_x = round(w / k);

x1 = 1 : max_x;
x2 = 1 : max_y;
[X1, X2] = meshgrid(x1, x2);

F = zeros(max_y, max_x); 
for i = 1 : size(mu, 1)
    F_ = mvnpdf([X1(:) X2(:)], mu(i, :) / k, Sigma);
    F = max(F, reshape(F_, length(x2), length(x1)));
end

F = imresize(F, [h, w]);