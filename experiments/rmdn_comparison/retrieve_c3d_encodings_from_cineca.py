"""
This script downloads all c3d encodings computed on cineca.
"""

import paramiko
import os
import argparse

from tqdm import tqdm


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--user")
    parser.add_argument("--password")
    args = parser.parse_args()

    assert args.host is not None, 'Please provide a correct host'
    assert args.user is not None, 'Please provide a correct username'
    assert args.password is not None, 'Please provide a correct password'

    # set up client
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password)
    sftp = ssh.open_sftp()

    sequences = range(1, 74+1)
    cur_seq = sequences[0]
    # send sequences
    while cur_seq <= sequences[-1]:
        try:
            print 'Retrieving sequence {}'.format(cur_seq)

            local_path = 'Z:/DATA/{:02d}/c3d_encodings/'.format(cur_seq)
            remote_path = '/gpfs/work/IscrC_DeepVD/dabati/DREYEVE/data/{:02d}/c3d_encodings/'.format(cur_seq)

            if not os.path.exists(local_path):
                os.makedirs(local_path)

            code_list = sftp.listdir(path=remote_path)

            for filename in tqdm(code_list):
                sftp.get(remote_path+filename, local_path+filename)

            print ''
            cur_seq += 1

        except paramiko.SSHException:
            # set up client again
            ssh = paramiko.SSHClient()
            ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(args.host, username=args.user, password=args.password)
            sftp = ssh.open_sftp()

            # redo sequence

    sftp.close()
    ssh.close()
