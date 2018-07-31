import codecs
import re
import sys
import subprocess
import argparse
import os
import json
from operator import itemgetter
# TODO: At some point of complexity, this will be needed
import kaldi_io


def which(program):
    """
    Equivalent to bash wich see https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def red(prompt_string):
    """"Append special chars to make command line prompt red"""
    return "\033[91m%s\033[0m" % prompt_string


def yellow(prompt_string):
    """"Append special chars to make command line prompt yellow"""
    return "\033[33m%s\033[0m" % prompt_string


def kaldi_call(command, arguments):
    """Call Kaldi binary through subprocess with minimal checks"""
    # Binary reachable
    assert which(command), \
        red("%s Kaldi binary not found, check ./path.sh" % command)
    # Make call
    popen = subprocess.Popen(
            "%s %s" % (command, arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True
    )
    popen.wait()
    output = popen.stdout.read()
    # Binary exited correctly
    if popen.returncode != 0:
        raise IOError(red(
            "\n\n%s call failed, returned this:\n\n%s" % (command, output)
        ))
    return output


def get_dim_vector(in_scp_file, in_scp):
    return sent2dim


def format2dict(kaldi_output):
    try:
        sentence2stats = dict([
            tuple(line.rstrip().split())
            for line in output.split('\n') if line
        ])
        sentence2stats = dict(
            zip(sentence2stats.keys(),
            map(int, sentence2stats.values()))
        )
    except ValueError:
        raise IOError(red(
            "Kaldi output could not be formatted, first 10 lines were "
            "these:\n\n%s" % '\n'.join(output.split('\n')[:10])
        ))
    return sentence2stats


def add_scp_data_to_input(in_data_json, in_scp, input_name, sent2shape, force):

    # SANITY CHECKS:
    # json and scp file list coincide
    json_utts = in_data_json['utts'].keys()
    scp_utts = in_scp.keys()

    # Add files to json
    new_json = {'utts': {}}
    for utt_name, utt_content in in_data_json['utts'].items():

        # Copy old content
        new_json['utts'][utt_name] = utt_content

        # Find latest input, increase counter
        existing_feature_index = None
        for input_index, input in enumerate(utt_content['input']):
            if input_name == input['name'] and not force:
                raise Exception(
                    "Asked to add scp data into feature %s, but this name is"
                    " already used. Use --force to overwrite" % input_name
                )
#                print(
#                    "%s: Will overwrite content of %s" %
#                    (yellow("WARNING"), input_index)
#                )
            existing_feature_index = input_index
            break

        if existing_feature_index:
            new_json['utts'][utt_name]['input'][input_index] = {
                u'feat': in_scp[utt_name],
                u'name': input_name,
                u'shape': sent2shape[utt_name]
            }

        else:
            new_json['utts'][utt_name]['input'].append({
                u'feat': in_scp[utt_name],
                u'name': input_name,
                u'shape': sent2shape[utt_name]
            })

    return new_json


def argument_parser(sys_argv):

    parser = argparse.ArgumentParser('Manipulate Kaldi/ESPNet data')
    # INPUTS
    parser.add_argument(
        '--in-scp-file',
        type=str,
        help='Input list containing paths of ark files'
    )
    parser.add_argument(
        '--in-ark-file',
        type=str,
        help='ark file (this is relevant for the Kaldi calls)'
    )
    parser.add_argument(
        '--ark-class',
        type=str,
        choices=['matrix', 'vector'],
        help='Type of ark file (this is relevant for the Kaldi calls)'
    )
    parser.add_argument(
        '--in-json-file',
        type=str,
        help='json format data that we wisth to act upon'
    )
    parser.add_argument(
        '--input-name',
        type=str,
        help='name of the feature we wil act upon'
    )
    parser.add_argument(
        '--in-pytorch-model',
        type=str,
        help='Pytorch model file'
    )
    # OUTPUTS / MODIFIERS
    parser.add_argument(
        '--out-json-file',
        type=str,
        help='optional json output file'
    )
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['add-scp-data-to-input', 'debug', 'check-json', 'check-scp'],
        help='Action to perform with the data'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        choices=[1],
        help='Verbosity level'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        default=False,
        help='Overwrite data if found'
    )
    args = parser.parse_args(sys_argv)
    if not args.out_json_file:
        args.out_json_file = args.in_json_file
    return args


# For future use: Code to load model
# import torch
# # Config
# idim, odim, train_args = pickle.load(open('model.conf'))
# # Model
# fid = open('model.acc.best')
# def cpu_loader(storage, location):
#     return storage
# model = torch.load(fid, map_location=cpu_loader)
# import ipdb;ipdb.set_trace(context=30)


if __name__ == '__main__':

    # Argument Handling
    args = argument_parser(sys.argv[1:])

    # Read scp
    if args.in_scp_file:
        with codecs.open(args.in_scp_file, 'r', 'utf-8') as fid:
            # TODO: Sanity check formatting content
            in_scp = dict([line.strip().split() for line in fid.readlines()])

    # Read ark
    if args.in_ark_file:
        assert args.ark_class, "--ark-class must be specified"
        if args.ark_class == 'matrix':
            ark_data = kaldi_io.read_mat(args.in_ark_file)
        else:
            ark_data = kaldi_io.read_vec_flt(args.in_ark_file)

    # Read json
    if args.in_json_file:
        with codecs.open(args.in_json_file, 'r', 'utf-8') as fid:
            in_data_json = json.load(fid)

    # Read pytorch model
    if args.in_pytorch_model:
        import torch
        def cpu_loader(storage, location):
            return storage
        with open(args.in_pytorch_model) as fid:
           in_pytorch_model = torch.load(fid, map_location=cpu_loader)

    if args.action == 'add-scp-data-to-input':

        # Requirements
        assert (
            args.in_scp_file and
            args.in_json_file and
            args.ark_class and
            args.input_name
        ), "Requires --in-scp-file --in-ark-class --input-name --in-json-file"

        # Read number of feature vectors
        if args.ark_class == 'matrix':

            try:
                sent2shape = {
                    utt_name: kaldi_io.read_mat(utt).shape
                    for utt_name, utt in in_scp.items()
                }
            except kaldi_io.UnknownMatrixHeader as e:
                print("\nAre you sure --ark-class is matrix?\n")
                raise e

            # Old method using Kaldi directly
#            # Read dimension of the vectors
#            output = kaldi_call(
#                'feat-to-dim',
#                '--print-args=false scp:%s ark,t:-' % args.in_scp_file
#            )
#            sent2dim = format2dict(output)
#
#            # Read number of frames of the vector
#            output = kaldi_call(
#                'feat-to-len',
#                '--print-args=false scp:%s ark,t:-' % args.in_scp_file
#            )
#            sent2len = format2dict(output)

        else:

            # Read one single vector to get size
            vector_shape = kaldi_io.read_vec_flt(in_scp.values()[0]).shape[0]
            sent2shape = {}
            for utt_name in in_scp.keys():
                sent2shape[utt_name] = [vector_shape]

        # Sanity Checks:
        assert 'utts' in in_data_json, \
            "Missing utts at top level of %s" % in_data_json

        # Sanity check: Same sentences
        json_utts = in_data_json['utts'].keys()
        scp_utts = in_scp.keys()
        assert len(json_utts) == len(scp_utts), (
            "JSON %s has %d utterances, scp %s has %d utterances, "
            "expected the same number" % (
                args.in_json_file,
                len(json_utts),
                args.in_scp_file,
                len(scp_utts)
            )
        )
        assert sorted(json_utts) == sorted(scp_utts), \
            "Utterance sets in json and scp differ"
        if json_utts != scp_utts:
            print(
                "%s: Utterance lists in json and scp differ in order" %
                yellow("WARNING")
            )

        # Create a new json by adding the new data
        new_json = add_scp_data_to_input(
            in_data_json,
            in_scp,
            args.input_name,
            sent2shape,
            args.force
        )

        # Write final json
        with codecs.open(args.out_json_file, 'w', 'utf-8') as fid:
            fid.write(json.dumps(
                new_json,
                indent=4,
                ensure_ascii=False,
                sort_keys=True
            ))
        if args.verbose:
            print(
                "Wrote to %s %d utterances" % (
                    args.out_json_file,
                    len(new_json['utts'])
                )
            )

    elif args.action == 'debug':
        # in_scp
        # in_data_json
        import ipdb;ipdb.set_trace(context=30)
        print("")

    elif args.action == 'check-json':

        assert args.in_json_file, "Need to provide --in-json-file"

        print("Check %d features" % len(in_data_json['utts']))

        working_folder = os.getcwd()
        for utt_name, utt in in_data_json['utts'].items():
            for feature in utt['input']:
                file_path = feature['feat'].split(':')[0]
                # Path checks
                if not os.path.isfile(file_path):
                    # Missing
                    import ipdb;ipdb.set_trace(context=30)
                    print("")
                elif feature['feat'][:len(working_folder)] != working_folder:
                    # Non local path
                    pass

                try:
                    # Try to read as matrix
                    feat_data = kaldi_io.read_mat(feature['feat'])
                except:
                    try:
                        # Try to read as vector
                        feat_data = kaldi_io.read_vec_flt(feature['feat'])
                    except:
                        import ipdb;ipdb.set_trace(context=30)
                        feat_data = kaldi_io.read_mat(file_path)
                        print("")

                if list(feat_data.shape) != feature['shape']:
                    import ipdb;ipdb.set_trace(context=30)
                    print("")


    elif args.action == 'check-scp':
        for utt in in_scp.values():
            try:
                # Try to read as matrix
                feat_data = kaldi_io.read_mat(utt)
            except:
                try:
                    # Try to read as vector
                    feat_data = kaldi_io.read_vec_flt(utt)
                except:
                    import ipdb;ipdb.set_trace(context=30)
                    print("")
