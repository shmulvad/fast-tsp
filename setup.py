import sys
import os
import io

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages

DIR = os.path.abspath(os.path.dirname(__file__))
DESCRIPTION = "A fast TSP solver with Python bindings"

# Import the README and use it as the long-description.
try:
    with io.open(os.path.join(DIR, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name="fast_tsp",
    version="0.0.1",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Søren Mulvad",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    cmake_install_dir="src/fast_tsp",
    include_package_data=True,
    extras_require={"test": ["pytest"]},
    python_requires=">=3.6",
)
