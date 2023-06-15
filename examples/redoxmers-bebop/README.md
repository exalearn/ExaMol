# Replicating Doan et al. 2020

This notebook shows how to replicate the results of [Doan et al. _Chem Mat_ (2020)](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.0c00768)
using example.
This examples uses DFT computations and, therefore, requires a supercomputer.
We configured it to run on Bebop.

## Set Up

Run `get_search_space.py` to get the search space.

Edit [`spec.py`](spec.py) to define how to launch Gaussian tasks


## Running the example

Navigate to this directory and then call

```
examol run spec.py:spec
```

It will output to `run` and eventually produce a file, `report.md`, that contains a summary of run.
