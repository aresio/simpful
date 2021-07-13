from setuptools import setup
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'simpful',
  packages = ['simpful'], # this must be the same as the name above
  version = '2.5.0',
  description = 'A user-friendly Python library for fuzzy logic',
  author = 'Marco S. Nobile',
  author_email = 'm.s.nobile@tue.nl',
  url = 'https://github.com/aresio/simpful', # use the URL to the github repo
  keywords = ['fuzzy logic', 'sugeno', 'mamdani', 'reasoner', 'python', 'modeling'], # arbitrary keywords
  license='LICENSE.txt',
  install_requires=[
        "numpy >= 1.12.0",
        "scipy >= 1.0.0",
        "requests",
    ],
  classifiers = ['Programming Language :: Python :: 3.7'],
  long_description=long_description,
  long_description_content_type='text/markdown',
)