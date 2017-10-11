clear; close all; clc;

% Add Space Variant Imaging toolbox to the matlab path
addpath(genpath('../external_libs/svistoolbox-1.0.5'))

% Add custom functions to apply multifovea
addpath(genpath('./multifovea'))

% Import python modules needed
cd('./python_interop')
utils = py.importlib.import_module('assessment_utils');
py.reload(utils) % useful if changed
cd('..')

% Sample a sequence and a start frame
python_tuple = utils.get_random_clip();
seq = double(python_tuple{1});
start = double(python_tuple{2});
is_acting = char(python_tuple{3});

% Get driver for sampled sequence
python_str = utils.get_driver_for_sequence(seq);
driver_id  = char(python_str);

% Sample an attentional map
maps = {'groundtruth', 'prediction', 'central_baseline'};
which_map = maps{randi(numel(maps))};

% Length of the sequence
n_frames = double(utils.n_frames);

% TODO do something here
for idx_to_load = start : start + n_frames
    disp(idx_to_load) 
end


