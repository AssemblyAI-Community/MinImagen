from minimagen.generate import load_minimagen, sample_and_save

training_directory = "training_20220806_153211"

minimagen = load_minimagen(training_directory)

captions = [
    'a happy dog',
    #'a big red house'
]

sample_and_save(minimagen, captions, sample_args={'cond_scale':3.}, sequential=True, directory='inference')