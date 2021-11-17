from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext = [Extension("py_smm_base", ["py_smm_base.pyx"])]


setup(name="py_smm_base",
      ext_modules=cythonize(ext))
