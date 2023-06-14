Simulate
========

The Simulate module defines restricted interfaces to computational chemistry codes.


Accuracy Levels
---------------

The core concept behind simulation in ExaMol is that there are few levels
The idea is to prioritize consistency of settings across computations
over flexibility in being able to run slightly-different computations in a workflow.

Each level maps to a different computational chemistry code using a specific set of parameters
that are validated for the `recipes <store.html#recipes>`_ available through ExaMol.

.. |ASESimulator| replace:: :class:`~examol.simulate.ase.ASESimulator`
.. list-table::
    :header-rows: 1

    * - Name
      - Interface
      - Code
      - Description
    * - xtb
      - |ASESimulator|
      - xTB
      - Tight binding using the GFN2-xTB parameterization
    * - guassian_[method]_[basis]
      - |ASESimulator|
      - Gaussian
      - Any method and basis set supported by Gaussian. Replace ``[method]`` and ``[basis]`` with desired settings
    * - cp2k_blyp_szv
      - |ASESimulator|
      - CP2K
      - Gaussian-Augmented Plane Wave DFT with a BLYP XC function and the SZV-GTH basis set
    * - cp2k_blyp_dzvp
      - |ASESimulator|
      - CP2K
      - Gaussian-Augmented Plane Wave DFT with a BLYP XC function and the DZVP-GTH basis set
    * - cp2k_blyp_tzvp
      - |ASESimulator|
      - CP2K
      - Gaussian-Augmented Plane Wave DFT with a BLYP XC function and the TZVP-GTH basis set


After selecting a level of accuracy, select the interface needed to run it.


The Simulator Interface
-----------------------

ExaMol provides workflow-compatible interfaces for common operations in quantum chemistry
through the :class:`~examol.simulate.base.BaseSimulator` interface.
Each Simulator implementation provides functions to compute the energy of a structure
and a function to perform a geometry optimization which take inputs and produce outputs
suitable for transmitting between computers.

Create a simulator interface by providing it first with any options needed to run on
your specific supercomputer.
An interface that will use CP2K could, for example, require a path to the scratch directory
and the mpiexec command used to launch it.

.. code-block:: python

    sim = ASESimulator(
        scratch_dir='cp2k-files',
        cp2k_command=f'mpiexec -n 8 --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                     '/path/to/exe/local_cuda/cp2k_shell.psmp',
    )

The interface can then run the energy computations or optimizations with CP2K.
Each computation returns a :class:`~examol.simulate.base.SimResult` object containing the
energy and structure of the outputs.

.. code-block:: python

    out_res, traj_res, _ = sim.optimize_structure(
        xyz=xyz,
        config_name='cp2k_blyp_dzvp',
        charge=0
    )
    solv_res, _ = sim.compute_energy(
        xyz=out_res.xyz,
        config_name='cp2k_blyp_dzvp',
        charge=1,
        solvent='acn'
    )

.. _levels:

Adding New Accuracy Levels
--------------------------

.. note::

    Work in Progress. Logan is working to make this easier (see `Issue #40 <https://github.com/exalearn/ExaMol/issues/40>`_)
