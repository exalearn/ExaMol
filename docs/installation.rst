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

Start by creating or activating the virtual environment in which you will run ExaMol then invoke pip

.. code-block:: shell

    pip install -e .

The default installation will install all _necessary_ packages but will skip some required for
some components, such as ``tensorflow`` and ``nfp`` for the :class:`~examol.scorer.nfp.NFPScorer`.
You may need to install these packages as you test ExaMol on your system.
