import subprocess
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
subprocess.call(["venv/Scripts/python", "train.py", "-test", "-ts", timestamp])
subprocess.call(["venv/Scripts/python", "inference.py", "-d", f"training_{timestamp}"])