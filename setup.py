import os
from setuptools import setup, find_packages

version_ns = {}
with open(os.path.join("examol", "version.py")) as f:
    exec(f.read(), version_ns)
version = version_ns['VERSION']

with open('requirements.txt') as f:
    install_requires = f.readlines()
    
with open('README.md') as f:
    long_desc = f.read()

setup(
    name='examol',
    version=version,
    packages=find_packages(include=['examol']),
    include_package_data=True,
    description='Molecular design on Exascale HPC',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    install_requires=install_requires,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    author="ExaLearn",
    author_email='lward@anl.gov',
    license="MIT License",
    url="https://github.com/exalearn/examol/"
)
