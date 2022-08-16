from setuptools import setup, find_packages

VERSION = '0.0.9'
DESCRIPTION = 'Minimal Imagen text-to-image model implementation.'
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

# Replace README local image paths with GitHub paths so images render on PyPi
LONG_DESCRIPTION = LONG_DESCRIPTION.replace("./images/", "https://github.com/AssemblyAI-Examples/MinImagen/raw/main/images/") 

#"Minimal Imagen text-to-image model implementation. See the [GitHub repo](https://github.com/AssemblyAI-Examples/MinImagen) or the [how-to build guide](www.assemblyai.com/blog/build-your-own-imagen-text-to-image-model/) for more details"

with open('requirements.txt', "r", encoding="utf-16") as f:
    required = f.read().splitlines()

# Setting up
setup(
    name="minimagen",
    version=VERSION,
    author="AssemblyAI",
    author_email="<ryan@assemblyai.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=required,
    keywords=[ 'imagen',
              'text-to-image',
              'diffusion model',
              'super resolution',
              'image generation',
              'machine learning',
              'deep learning',
              'pytorch',
              'python'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
