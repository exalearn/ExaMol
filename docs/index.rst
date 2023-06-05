.. ExaMol documentation master file, created by
   sphinx-quickstart on Wed Mar  8 16:51:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ExaMol
======

ExaMol combines AI, quantum chemistry, and supercomputers to design molecules as fast as possible.

Build an ExaMol application by defining the properties to be optimized,
the AI models and optimization strategy,
and computation resources as a Python object.

Once defined, execute by calling it from the command line:

.. code-block:: shell

   examol run examples/redoxmers/spec.py:spec

Begin with our `Quickstart <quickstart.html>`_ to learn how to create the specification file,
then continue with the following section about the components of an ExaMol application
if your science requires new capabilities.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   components/index
   api/examol
