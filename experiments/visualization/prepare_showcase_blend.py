"""
Script for preparing showcase videos.
A bunch of predictions from the test set are showcased.
Failure cases are included at the end.
"""
import os
import pandas as pd
import tempfile
from os.path import join
from random import shuffle


blend_videos_dir  = '/majinbu/public/DREYEVE/blended_videos'  # blended videos location
out_showcase_path = 'showcase.avi'                            # where to save output showcase video


def prepare_ffmpeg_command(case, seq_num, start_sec, end_sec, verbose=True):
    """
    Prepare ffmpeg command to extract a subsequence from a longer prediction video. 
    """
    cur_video = join(blend_videos_dir, '{}.avi'.format(seq_num))

    ss = '-ss {}'.format(start_sec)                    # start second
    i = '-i "{}"'.format(cur_video)                    # input file path
    t = '-t {}'.format(int(end_sec) - int(start_sec))  # how many seconds to keep
    r = '-r 25'                                        # frame rate
    quality = '-qscale:v 2'                            # quality in 1-31, 1 being highest quality
    str_case = '' if case == 'success' else 'FAILURE CASE - '
    drawtext = '"drawtext=x=50: y=950: fontsize=75: fontcolor=white:fontfile=/usr/share/texlive/texmf-dist/fonts/truetype/public/opensans/OpenSans-Regular.ttf: text=\'{}Run {}, seconds [{}-{}]\'"'.format(
        str_case, seq_num, start_sec, end_sec)
    out_path = join(temp_dir, '{}_{}_{}.avi'.format(seq_num, start_sec, end_sec))
    args = ['ffmpeg', ss, i, t, r, quality, '-vf', drawtext, out_path]
    command = ' '.join(args)
    if verbose:
        print(command)
    return command, out_path


if __name__ == '__main__':

    # File containing some nice to showcase sequences
    annotation_file = 'subseq2showcase.csv'
    data_frame = pd.read_csv(annotation_file)

    cases = ['success', 'failure']
    out_paths_dict = {k: [] for k in cases}

    # All the following processing is performed in a temporary directory. When the scripts
    #  ends, only the output video in `out_showcase_path` is saved permanently.
    with tempfile.TemporaryDirectory() as temp_dir:

        for current_case in cases:
            data_success = data_frame[data_frame['case'] == current_case]

            # Save all interesting chunks in a temporary directory
            for idx, row in data_success.iterrows():

                # Format ffmpeg command accordingly to current video sequence
                command, out_path = prepare_ffmpeg_command(case=current_case,
                                                           seq_num=row['sequence'],
                                                           start_sec=row['start_sec'],
                                                           end_sec=row['end_sec'])
                os.system(command)

                # Store output path for later concatenation
                out_paths_dict[current_case].append(out_path)

        # The starting video must be the beginning of run 40 (starting from traffic light)
        video_to_start_with = join(temp_dir, '40_0_6.avi')
        start_40_idx = out_paths_dict['success'].index(video_to_start_with)
        out_paths_dict['success'].pop(start_40_idx)

        # Shuffle both success and failure case
        [shuffle(v) for (k, v) in out_paths_dict.items()]

        # Create temporary list of file paths to be fed to ffmpeg to concatenate everything together
        dump_video_list = join(temp_dir, 'list.txt')
        video_paths_list = [video_to_start_with] + out_paths_dict['success'] + out_paths_dict['failure']
        with open(dump_video_list, mode='a') as f:
            [f.write('file \'{}\'\n'.format(vid)) for vid in video_paths_list]

        # Merge all chunks together
        command = 'ffmpeg -y -f concat -safe 0 -i {} -c copy {}'.format(dump_video_list, out_showcase_path)
        os.system(command)
