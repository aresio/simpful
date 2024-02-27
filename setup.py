from setuptools import setup
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'simpful',
  packages = ['simpful'], # this must be the same as the name above
  version = '2.12.0',
  description = 'A user-friendly Python library for fuzzy logic',
  author = 'Marco S. Nobile',
  author_email = 'marco.nobile@unive.it',
  url = 'https://github.com/aresio/simpful', # use the URL to the github repo
  keywords = ['fuzzy logic', 'sugeno', 'mamdani', 'reasoner', 'python', 'modeling'], # arbitrary keywords
  license='LICENSE.txt',
  install_requires=[
        "numpy >= 1.12.0",
        "scipy >= 1.0.0",
    ],
  extras_require={
        "plotting": ["matplotlib>=3.5.1", "seaborn>=0.11.2"],
    },
  classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Academic Free License (AFL)",
    "Operating System :: OS Independent",],
  long_description=long_description,
  long_description_content_type='text/markdown',
)
