clear; close; clc;

m = 1; mm = m / 1000; um = m/1000000;

pixel_size = 1.55 * um;
image_width = 1920;
sensor_pixels = 12 * 1000000;  % 12 * 10^6
sensor_width = pixel_size * sensor_pixels;  % meters

% conversions
meters_to_pixels = 1 / sensor_width * image_width;
pixels_to_meters = 1 / meters_to_pixels;

%% Plot the solera bound as a function of f at different focal lengths ratios ||x_2||/h.

% Plot at different ratios between ||x_2|| and h
x2_h = 1:7;

% X axis, focal lenghts
f_lengths = (0:1:60) * mm;  % measured in meters
f_lengths = f_lengths * meters_to_pixels;  % measured in pixels

% Loop over ||x_2||/h ratios
fig = figure();
for r_idx=1:numel(x2_h)
    r = x2_h(r_idx);
    
    b_px = abs((2*f_lengths)/(1+r));
    
    h(r_idx) = plot(f_lengths * pixels_to_meters / mm, b_px, 'DisplayName', sprintf('%.02f', r));
    hold on;
end
plot([50, 50], [0, 25], 'k--', 'linewidth', 2, 'DisplayName', '');
lgd = legend(h, 'Location', 'northeast', 'Orientation', 'vertical');
title(lgd,'||x_2||/h')
ylabel('px'); xlabel('focal length (mm)');
title('projection error on camera A');
grid on

return
%% Plot the solera bound as a function of ||x_2||/h at different focal lengths f.

m = 1; mm = m / 1000;

% Plot at different focal lenghts
f_lengths = ([40, 45, 50, 55, 60]) * mm;

% X axis, the ratio between ||x_2|| and h
x2_h = 0:0.5:7;

% Loop over focal lengths
fig = figure();
for f_idx=1:numel(f_lengths)
    f = f_lengths(f_idx);
    f = f * meters_to_pixels
    
    b_px = abs((2*f)./(1+x2_h));
    
    h(f_idx) = plot(x2_h, b_px, 'DisplayName', sprintf('%.02f', f* pixels_to_meters / mm));
    hold on
    
end
legend('show');
lgd = legend(h, 'Location', 'northeast', 'Orientation', 'vertical');
title(lgd,'f')
ylabel('px'); xlabel('||x_2||/h');
title('projection error on camera A');