import sys
import os
import argparse
import codecs
import numpy as np


def read_file(file_path):
    with codecs.open(file_path, 'r') as fid:
        lines = [line.strip() for line in fid.readlines()]
    return lines


def write_file(file_path, lines):
    with codecs.open(file_path, 'w') as fid:
        for line in lines:
            fid.write('%s\n' % line)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Manipulate Kaldi/ESPNet data')
    # INPUTS
    parser.add_argument('--in-data-folder', type=str,required=True)
    parser.add_argument('--out-data-folder', type=str,required=True)
    parser.add_argument('--subsample-factor', type=int, default=0.01)
    parser.add_argument('--min-size', type=int, default=80)
    args = parser.parse_args(sys.argv[1:])

    if not os.path.isdir(args.out_data_folder):
        os.makedirs(args.out_data_folder)

    for sset in os.listdir(args.in_data_folder):

        # spk2gender  spk2utt  text  utt2spk  wav.scp
        sset_path = os.sep.join([args.in_data_folder, sset])
        out_sset_path = os.sep.join([args.out_data_folder, sset])

        # Need to be subsampled
        wavs = read_file("%s/wav.scp" % (sset_path))
        text = read_file("%s/text" % (sset_path))
        utt2spk = read_file("%s/utt2spk" % (sset_path))

        # Get Subsample
        nr_files = len(wavs)
        new_nr_files = np.max([
            int(np.ceil(args.subsample_factor*nr_files)), args.min_size
        ])
        new_indices = np.random.choice(range(nr_files), new_nr_files)
        # subsample
        new_wavs = [wavs[idx] for idx in new_indices]
        new_text = [text[idx] for idx in new_indices]
        new_utt2spk = [utt2spk[idx] for idx in new_indices]

        # Do not need to be subsampled
        spk2utt = read_file("%s/spk2utt" % (sset_path))
        spk2gender = read_file("%s/spk2gender" % (sset_path))

        # Write
        if not os.path.isdir(out_sset_path):
            os.makedirs(out_sset_path)
        write_file("%s/wav.scp" % (out_sset_path), new_wavs)
        write_file("%s/text" % (out_sset_path), new_text)
        write_file("%s/utt2spk" % (out_sset_path), utt2spk)
        write_file("%s/spk2utt" % (out_sset_path), spk2utt)
        write_file("%s/uspk2gender" % (out_sset_path), spk2gender)
        print("%s" % out_sset_path)
