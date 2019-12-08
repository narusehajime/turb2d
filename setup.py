# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='turb2d',
      version='0.1.0',
      description='2D shallow water model for turbidity currents',
      long_description=readme,
      author='Hajime Naruse',
      author_email='naruse@kueps.kyoto-u.ac.jp',
      url='https://github.com/narusehajime/turb2d.git',
      license=license,
      install_requires=['numpy', 'landlab', 'matplotlib', 'gdal'],
      packages=find_packages(exclude=('tests', 'docs')))
