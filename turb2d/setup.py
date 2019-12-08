import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext = Extension("cip",
                sources=["cip.pyx"],
                include_dirs=['.', np.get_include()])
setup(name="cip", ext_modules=cythonize([ext]))
