import subprocess
from datetime import datetime

# Get timestamp for training
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Run training on small test Imagen
subprocess.call(["python", "train_pokemon.py", "-test", "-ts", timestamp])
# image_size = '64'
# batch_size = '1'
# subprocess.call(["python", "train_pokemon.py", "-s", image_size, "-b", batch_size, "-test", "-ts", timestamp])

# subprocess.call(["python", "train.py", "-test", "-ts", timestamp])

# Use small test Imagen to generate image
subprocess.call(["python", "inference.py", "-d", f"training_{timestamp}"])  # noqa: E501