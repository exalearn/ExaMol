Quickstart
==========

An ExaMol application is divided into a _Thinker_ which defines which tasks to run
and a _Doer_ which executes them on HPC resources.
Your task is to define the Thinker and Doer by a Python specification object.

The specification object, :class:`~examol.specify.ExaMolSpecification`,
describes what thinker is seeking to optimize,
how it will select calculations,
what those computations are,
and a description of the resources available to it.

We'll go through each option briefly here,
and link out to pages that describe the full options available for each.

Starting Data
-------------


