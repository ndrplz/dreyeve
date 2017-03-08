import paramiko
from glob import glob
import os
from tqdm import tqdm
import argparse


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
            print 'Sending sequence {}'.format(cur_seq)

            local_path = 'Z:/DATA/{:02d}/optical_flow/*.png'.format(cur_seq)
            remote_path = '/gpfs/work/IscrC_DeepVD/dabati/DREYEVE/data/{:02d}/optical_flow/'.format(cur_seq)

            img_list = glob(local_path)

            for im in tqdm(img_list):
                filename = os.path.basename(im)
                sftp.put(im, remote_path+filename)
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
