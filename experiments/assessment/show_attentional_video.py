"""
This script draws an attentional video, asks questions to 
the subject and saves its answers.
"""

import numpy as np
import skvideo.io

from os.path import join, exists
import cv2

from create_attentional_videos import output_root as video_root
from create_attentional_videos import output_txt as input_txt

from questions import ask_question_1, ask_question_2


# parameters
output_file = 'assessment_answers.txt'
videos_each_subject = 5
videos_to_skip = 0


def get_subject_id():
    """
    Reads the answer file and provides a new subject id.
    
    Returns
    -------
    str:
        a new subject id like: `subject_004`
    """

    if not exists(output_file):
        subject_id = 0
    else:
        with open(output_file, mode='r') as f:
            subjects = [int(l.split('\t')[0][-3:]) for l in f.readlines()]
        subject_id = np.max(subjects) + 1

    return 'subject_{:03d}'.format(subject_id)


def get_random_video():
    """
    Reads the file containing the list of videos
    and randomly samples one of them, returning related information.
    
    Returns
    -------
    list
        a list of video parameters (filename, attentional_behavior, dreyeve_sequence, frame_start, frame_end)
    """

    with open(input_txt, mode='r') as f:
        lines = [l.strip() for l in f.readlines()]

    video_line = np.random.choice(lines)
    video_params = video_line.split('\t')

    return video_params


def log_on_file(subject_id, video_filename, perceived_safeness, turing_guess, attentional_behavior,
                seq, start, end, is_acting, color_slope, spatial_slope):
    """
    Logs the answers of a subject to the text file.
    
    Parameters
    ----------
    subject_id: str
        the id of the subject.
    video_filename: str
        the filename of the video
    perceived_safeness: int
        the level of perceived safeness of attentional maps.
    turing_guess: str
        guess of the subject among [`Human`, `AI`].
    attentional_behavior: str
        the actual source of attentional maps among [`groundtruth`, `prediction`, `central_baseline`].
    seq: int
        the number of the dreyeve sequence.
    start: int
        the starting frame in the dreyeve sequence.
    end: int
        the ending frame in the dreyeve sequence.
    """

    with open(output_file, 'a') as f:
        line = [subject_id, video_filename, perceived_safeness, turing_guess, attentional_behavior,
                seq, start, end, is_acting, color_slope, spatial_slope]
        f.write(('{}\t'*len(line)).format(*line).rstrip())
        f.write('\n')


def main():
    """ Main function """

    # get an id for the new subject
    subject_id = get_subject_id()

    for i in range(0, videos_each_subject):
        # get a random video
        video_filename, driver, attentional_behavior, seq, start, end, is_acting, color_slope, spatial_slope = get_random_video()

        # display video
        video_path = join(video_root, video_filename)
        frames = skvideo.io.FFmpegReader(video_path).nextFrame()  # generator

        for frame in frames:
            frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # TODO pls remove this library I hate it.

            cv2.waitKey(1000 // 25)

        # ask questions
        perceived_safeness = ask_question_1()
        turing_guess = ask_question_2()

        # write line on file
        if i >= videos_to_skip:
            log_on_file(subject_id, video_filename, perceived_safeness, turing_guess, attentional_behavior,
                        seq, start, end, is_acting, color_slope, spatial_slope
                        )


# entry point
if __name__ == '__main__':
    main()
