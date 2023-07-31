# Replicating Doan et al. 2020

This notebook shows how to replicate the results of [Doan et al. _Chem Mat_ (2020)](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c00768)
using example.
This examples uses DFT computations and, therefore, requires a supercomputer.
We configured it to run on Bebop.

## Running the example

Navigate to this directory and then call

```
examol run spec.py:spec
```

It will output to `run` and eventually produce a file, `report.md`, that contains a summary of run.

## Notes

There are a few limitations to be aware of for now:

1. Every calculation runs on a single node, regardless of size
1. Large molecules use the ASE optimizer, smaller ones use Gaussian's
