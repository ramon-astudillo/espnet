# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 3.0  (http://www.apache.org/licenses/LICENSE-2.0)

import codecs
import argparse
from backtranslation.cleaners import english_cleaners

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='text to be cleaned')
    args = parser.parse_args()
    with codecs.open(args.text, 'r', 'utf-8') as fid:
        for line in fid.readlines():
            id, content, _ = line.split("|")
            clean_content = english_cleaners(line.rstrip())
            print("%s %s" % (id, clean_content))
