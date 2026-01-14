# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "owrf1d._core",
        ["src/owrf1d/_core.pyx"],
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
        annotate=True,
    )
)

# pip install -e ".[dev,examples]"
# python -m pip install -v -e .
# python -m build
# python -m twine check dist/*
