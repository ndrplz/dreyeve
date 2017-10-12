function [ attention_map ] = unzip_and_load_npz(npz_file_path)
%UNZIP_AND_LOAD_NPZ Unzip and load attention map stored in npz format
%   Unzip a file stored in `.npz` format into the temp directory.
%   Temp file `.npy` is deleted after being loaded
    
    % Once unzipped all file have this name
    npy_file_path = fullfile(tempdir, 'arr_0.npy');
    
    % Be sure `.npy` is not already present in tempdir
    if exist(npy_file_path, 'file')
        delete(npy_file_path)
    end
    
    % Unzip `.npz` into tempdir
    unzip(npz_file_path, tempdir);
    
    % Load `.npy` file (all have the same name)
    attention_map = readNPY(npy_file_path);
   
    % Be sure to delte `.npy` file from tempdir
    if exist(npy_file_path, 'file')
        delete(npy_file_path)
    end

end

