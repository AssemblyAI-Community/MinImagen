from argparse import ArgumentParser

from minimagen.generate import load_minimagen, sample_and_save

# Command line argument parser
parser = ArgumentParser()
parser.add_argument("-c", "--CAPTIONS", dest="captions", help="Single caption to generate for or filepath for .txt "
                                                              "file of captions to generate for or ", default=None, type=str)
parser.add_argument("-d", "--TRAINING_DIRECTORY", dest="training_directory", help="Directory generated from training to use for inference", type=str)
args = parser.parse_args()

minimagen = load_minimagen(args.training_directory)

if args.captions is None:
    print("\nNo caption supplied - using the default of \"a happy dog\".\n")
    captions = ['a happy dog']
elif not args.captions.endswith(".txt"):
    captions = [args.captions]
elif args.captions.endswith(".txt"):
    with open(args.captions, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            lines[idx] = line[:-1] if line.endswith('\n') else line
            list(filter(lambda x: x, lines))
else:
    raise ValueError("Please input a valid argument for --CAPTIONS")

sample_and_save(minimagen, captions, sample_args={'cond_scale':3.})