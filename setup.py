from setuptools import setup, find_packages
from os import path

# Read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='simpful',
    packages=find_packages(),  # Automatically find packages
    version='2.13.0',  # Updated version for a non-breaking change
    description='A user-friendly Python library for fuzzy logic',
    author='Marco S. Nobile',
    author_email='marco.nobile@unive.it',
    url='https://github.com/aresio/simpful',  # use the URL to the github repo
    keywords=['fuzzy logic', 'sugeno', 'mamdani', 'reasoner', 'python', 'modeling'],  # arbitrary keywords
    license='LICENSE.txt',
    install_requires=[
        "numpy >= 1.12.0",
        "scipy >= 1.0.0",
        "pandas >= 1.0.0",
        "aiohttp == 3.9.5",
        "openai == 0.28.0",
        "requests == 2.32.2",
        "scikit-fuzzy == 0.4.2",
        "tqdm == 4.66.4",
    ],
    extras_require={
        "plotting": ["matplotlib>=3.5.1", "seaborn>=0.11.2"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Academic Free License (AFL)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
