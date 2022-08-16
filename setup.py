from setuptools import setup, find_packages

VERSION = '0.0.5'
DESCRIPTION = 'Minimal Imagen text-to-image model implementation.'
with open("README.md") as f:
    LONG_DESCRIPTION = f.read() 

#"Minimal Imagen text-to-image model implementation. See the [GitHub repo](https://github.com/AssemblyAI-Examples/MinImagen) or the [how-to build guide](www.assemblyai.com/blog/build-your-own-imagen-text-to-image-model/) for more details"

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Setting up
setup(
    name="minimagen",
    version=VERSION,
    author="AssemblyAI",
    author_email="<ryan@assemblyai.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
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
