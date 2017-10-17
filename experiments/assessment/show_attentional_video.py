"""
This script draws an load a precomputed attentional video,
asks questions to the subject and saves its answers.
"""


import cv2
import numpy as np
import skvideo.io
from os.path import join, exists
from questions import ask_question_1
from questions import ask_question_2


class VideoSignature:
    """
    This class collects all information available for a certain assessment video.
    """
    def __init__(self, **kwargs):

        # Add all members loaded from file
        self.__dict__.update(kwargs)

        # Add numeric version of these parameters in case we need
        self.seq           = int(self.seq_str)
        self.start_frame   = int(self.start_frame_str)
        self.end_frame     = int(self.end_frame_str)
        self.sequence_area = float(self.sequence_area_str)
        self.count_acting  = int(self.count_acting_str)

    def __str__(self):
        """
        Override string representation of a VideoSignature to ease readability when logging
        """
        return '{},{},{},{},{},{},{},{},{}'.format(self.video_filename, self.driver_id, self.which_map, self.seq_str,
                                                   self.start_frame_str, self.end_frame_str, self.is_acting,
                                                   self.sequence_area_str, self.count_acting_str)


def get_subject_id(answers_file, separator=','):
    """
    Reads the answer file and provides a new subject id.
    
    Returns
    -------
    str:
        a new subject id like: `subject_004`
    """

    if not exists(answers_file):
        subject_id = 0
    else:
        with open(answers_file, mode='r') as f:
            subjects = [int(l.split(separator)[0][-3:]) for l in f.readlines()]
        subject_id = np.max(subjects) + 1

    return 'subject_{:03d}'.format(subject_id)


def get_random_video(video_signatures_file, line_separator):
    """
    Reads the file containing the list of videos
    and randomly samples one of them, returning related information.
    
    Returns
    -------
    VideoSignature
        VideoSignature object which contains all information on assessment video
    """

    with open(video_signatures_file, mode='r') as f:
        lines = [l.strip() for l in f.readlines()]

    # Load and parse a random line in the video signature file
    video_line = np.random.choice(lines)
    video_params = video_line.split(line_separator)

    # Create and return a VideoSignature object with all information we need for this video
    signature_dict = {'video_filename': video_params[0],
                      'driver_id': video_params[1],
                      'which_map': video_params[2],
                      'seq_str': video_params[3],
                      'start_frame_str': video_params[4],
                      'end_frame_str': video_params[5],
                      'is_acting': video_params[6],
                      'sequence_area_str': video_params[7],
                      'count_acting_str': video_params[8]}
    return VideoSignature(**signature_dict)


def log_on_file(output_file, subject_id, video_signature, subject_answers):
    """
    Logs the answers of a subject to the text file.
    
    Parameters
    ----------
    output_file: str
        path to the output file
    subject_id: str
        identifier of the subject
    video_signature: VideoSignature
        signature of the sampled video
    subject_answers: list
        subject's answers [perceived_safeness, turing_guess]
    """

    with open(output_file, 'a') as f:

        perceived_safeness, turing_guess = subject_answers

        line = '{},{},{},{}\n'.format(subject_id, video_signature, perceived_safeness, turing_guess)
        f.write(line)


def main():
    """ Main function """

    video_root = '/majinbu/public/DREYEVE/QUALITY_ASSESSMENT_VIDEOS_MATLAB'
    video_signatures_file = join(video_root, 'videos.txt')

    # Experiment parameters
    answers_file = 'assessment_answers.txt'
    videos_each_subject = 5
    videos_to_skip = 0

    # ID of the current subject
    subject_id = get_subject_id(answers_file)

    for i in range(0, videos_each_subject):

        # Get a random video signature from pre-processed videos
        video_signature = get_random_video(video_signatures_file, line_separator=';')

        # Display video
        video_path = join(video_root, video_signature.video_filename)
        frames = skvideo.io.FFmpegReader(video_path).nextFrame()  # generator
        for frame in frames:
            frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1000 // 25)

        # Ask questions
        perceived_safeness = ask_question_1()
        turing_guess = ask_question_2()

        # Log on file the current answer
        if i >= videos_to_skip:
            log_on_file(answers_file, subject_id, video_signature, subject_answers=[perceived_safeness, turing_guess])


# Entry point
if __name__ == '__main__':
    main()
