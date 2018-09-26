from distutils.core import setup
setup(
  name = 'simpful',
  packages = ['simpful'], # this must be the same as the name above
  version = '1.0.3',
  description = 'A simple Fuzzy Logic library',
  author = 'Marco S. Nobile',
  author_email = 'nobile@disco.unimib.it',
  url = 'https://github.com/aresio/simpful', # use the URL to the github repo
  keywords = ['fuzzy logic', 'sugeno', 'reasoner', 'python'], # arbitrary keywords
  license='LICENSE.txt',
  install_requires=[
        "numpy >= 1.12.0",
        "scipy >= 1.0.0"
    ],
  classifiers = ['Programming Language :: Python :: 2.7'],
)