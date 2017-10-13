function save_video_line_on_log_file( output_logfile, video_signature)
%SAVE_VIDEO_LINE_ON_LOG_FILE Saves a video signature on the a file, in
% append mode

fileID = fopen(output_logfile, 'a');
fprintf(fileID, video_signature);
fprintf(fileID, '\n');
fclose(fileID);

end

