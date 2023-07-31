# Redoxmer Design Example

This example shows how to use ExaMol to find molecules with optimized redox potentials.
For simplicity, we use [xTB](https://xtb-docs.readthedocs.io/en/latest/contents.html) to compute the redox potentials
so that this whole example can run on your laptop.

## Running the example

Navigate to this directory and then call

```
examol run spec.py:spec
```

It will output to `run` and eventually produce a file, `report.md`, that contains a summary of run.

## Set Up

The `generate_database.py` script produces the initial database and search space for the molecules.
These files should be available along with this README, but feel free to modify the script to change
the search problem.
