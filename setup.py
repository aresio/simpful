from distutils.core import setup
setup(
  name = 'simpful',
  packages = ['simpful'], # this must be the same as the name above
  version = '1.4.8',
  description = 'A simple Fuzzy Logic library',
  author = 'Marco S. Nobile',
  author_email = 'm.s.nobile@tue.nl',
  url = 'https://github.com/aresio/simpful', # use the URL to the github repo
  keywords = ['fuzzy logic', 'sugeno', 'reasoner', 'python'], # arbitrary keywords
  license='LICENSE.txt',
  install_requires=[
        "numpy >= 1.12.0",
        "scipy >= 1.0.0"
    ],
  classifiers = ['Programming Language :: Python :: 2.7', 'Programming Language :: Python :: 3.7'],
  long_description='simpful is a Python library for fuzzy logic reasoning, \
  designed to provide a simple and lightweight API, as close as possible \
  to natural language.',
)