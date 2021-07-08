from setuptools import setup, find_packages

setup(
  name = 'halonet-pytorch',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'HaloNet - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/halonet-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
