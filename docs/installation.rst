Installation
============

ExaMol can be installed as a normal Python package.
We recommend installing that package inside of a virtual environment,
and prefer to use Anaconda to build the environment because it can install
non-Python dependencies (e.g., MongoDB).

The first step to installing ExaMol is to download it from GitHub.
For example, create an updatable copy of the repository by

.. code-block:: shell

    git clone git@github.com:exalearn/ExaMol.git

Recommended: Anaconda
---------------------

The ``envs`` folder of ExaMol contains environment files suitable for different computers.

The best one to start with installs CPU versions of all software:

.. code-block:: shell

    conda env create --file envs/environment-cpu.yaml

Installation with Pip
---------------------

Start by creating or activating a virtual environment for ExaMol then invoke pip

.. code-block:: shell

    pip install -e .

The default installation will install all _necessary_ packages but will skip some required for
some components, such as ``tensorflow`` and ``nfp`` for the :class:`~examol.scorer.nfp.NFPScorer`.

Running ExaMol on Mac
---------------------

The [environment-macos.yml](https://github.com/exalearn/ExaMol/blob/main/envs/environment-macos.yml) is 
designed to run on OS X. It will not run all features (e.g., xTB computations and Tensorflow-based models)
but is enough to test many features

.. code-block:: shell
    conda env create --file envs/environment-cpu.yaml

Modifying an Installation
-------------------------

ExaMol is designed so that difficult-to-install packages are not necessary unless specific features are needed.
For example, PyTorch need not be installed unless BOTorch selectors
or machine learning models based on Torch are required.
ExaMol will raise error messages that specify which optional dependencies are not met,
and the list of the dependencies is in ``pyproject.yaml``.

A best practice for including optional dependencies dependencies is to modify the Anaconda environment file
to include the versions best for specific hardware.
The repository provides examples for `CPU-only systems <https://github.com/exalearn/ExaMol/blob/main/envs/environment-cpu.yml>`_,
`CUDA 11.8 <https://github.com/exalearn/ExaMol/blob/main/envs/environment-cuda118.yml>`_,
and the `Polaris supercomputer <https://github.com/exalearn/ExaMol/blob/main/envs/environment-polaris.yml>`_ as examples.
Note how each differ in which version of PyTorch is installed and whether we use
the Anaconda version (which installs with CUDA libraries) or the PyPI version (which relies on system-provided versions).
