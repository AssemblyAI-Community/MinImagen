from argparse import ArgumentParser
from minimagen.generate import load_minimagen, sample_and_save

# Command line argument parser
parser = ArgumentParser()
parser.add_argument("-c", "--CAPTIONS", dest="CAPTIONS", help="Single caption to generate for or filepath for .txt "
                                                              "file of captions to generate for", default=None, type=str)
parser.add_argument("-d", "--TRAINING_DIRECTORY", dest="TRAINING_DIRECTORY", help="Training directory to use for inference", type=str)
args = parser.parse_args()

minimagen = load_minimagen(args.TRAINING_DIRECTORY)

if args.CAPTIONS is None:
    print("\nNo caption supplied - using the default of \"a happy dog\".\n")
    captions = ['a happy dog']
elif not args.CAPTIONS.endswith(".txt"):
    captions = [args.CAPTIONS]
elif args.CAPTIONS.endswith(".txt"):
    with open(args.CAPTIONS, 'r') as f:
        lines = f.readlines()
    captions = [line[:-1] if line.endswith('\n') else line for line in lines]
else:
    raise ValueError("Please input a valid argument for --CAPTIONS")

# Can supply a training dictionary to load from for inference
sample_and_save(captions, training_directory=args.TRAINING_DIRECTORY, sample_args={'cond_scale':3.})

# Otherwise, can supply a MinImagen instance itself. In this case, information about the instance will not be saved.
# sample_and_save(captions, minimagen=minimagen, sample_args={'cond_scale':3.})