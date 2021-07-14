import shutil
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Create json of folder tree.')
parser.add_argument('--rootdir', type=str,    help="Root directory.")
parser.add_argument('--outputfilename', type=str,    help="Output file name")
args = parser.parse_args()
root = args.rootdir
outputfilename = args.outputfilename
outputpath = os.path.join(os.getcwd(), outputfilename)


with open(outputpath, 'w') as f:
    for path, subdirs, files in os.walk(root):
        for name in files:
            print(name)
            f.write('{}\n'.format(os.path.join(path, name)))
