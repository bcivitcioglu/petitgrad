# setup.py
from setuptools import setup, find_packages

setup(
    name="petitgrad",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    tests_require=[
        "unittest",
    ],
    author="Burak Ç.",
    author_email="bcivitcioglu@gmail.com",
    description="A tiny or petit autograd engine with matrix operations, based on Andrej Karpathy's micrograd.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/bcivitcioglu/petitgrad",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
