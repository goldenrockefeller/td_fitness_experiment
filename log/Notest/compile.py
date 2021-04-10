from setuptools import setup
from Cython.Build import cythonize
import sys

if len(sys.argv) == 1:
    sys.argv.append('build_ext')
    sys.argv.append('--inplace')
else:
    raise NotImplementedError


setup(
    ext_modules
        = cythonize(
            ["td_fitness.py"],
            annotate = True,
            compiler_directives={'language_level': 3},
        ),
    zip_safe=False,
)
