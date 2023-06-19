# Replicating Doan et al. 2020

This notebook shows how to replicate the results of [Doan et al. _Chem Mat_ (2020)](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c00768)
using example.
This examples uses DFT computations and, therefore, requires a supercomputer.
We configured it to run on Bebop.

## Set Up

This example requires running a yet-unreleased version of ASE that supports Gaussian with >50 atoms. Install by

`pip install git+https://gitlab.com/argon214/ase.git@gaussian`

## Running the example

Navigate to this directory and then call

```
examol run spec.py:spec
```

It will output to `run` and eventually produce a file, `report.md`, that contains a summary of run.

## Notes

There are a few limitations to be aware of for now:

1. We can only run a single Gaussian calculation per block of nodes
