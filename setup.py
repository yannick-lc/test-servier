import os
from setuptools import setup

setup(
    name = "molecule",
    version = "0.0.1",
    author = "Yannick Le Cacheux",
    description = "Technical test of Yannick Le Cacheux for Servier.",
    url = "https://github.com/yannick-lc/test-servier",
    packages=['molecule'],
    entry_points={'console_scripts': ['servier=molecule.main:main']}
)