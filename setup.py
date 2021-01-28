from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Intended Audience :: Developers',
  'Intended Audience :: Science/Research',
  'Topic :: Scientific/Engineering :: Visualization',
  'Operating System :: OS Independent',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]

setup(
  name='kds',
  version='0.1.1',
  description='An intuitive library to plot evaluation metrics.',
  long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
  long_description_content_type='text/markdown',
  url='https://github.com/tensorbored/kds',
  author='Prateek Sharma',
  author_email='s.prateek3080@gmail.com',
  license='MIT License', 
  platforms='any',
  classifiers=classifiers,
  keywords='keytodatascience', 
  packages=['kds'],
  install_requires=[
    'matplotlib>=1.4.0',
    'pandas>=0.20.0',
    'numpy>=1.12.0'
  ],

  # entry_points={
  #   "console_scripts": [
  #       "kds=kds.__main__:main", # if library name is keytodatascience then we can shorten it using this
  #     ]
  #   }
)