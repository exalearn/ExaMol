# Check Accuracy of Quantum Chemistry

Benchmark different configurations for quantum chemistry codes and establish initial training sets.

The `run_many` scripts perform structure optimizations and solvation energy computations for all molecules in a provided list.

Once complete, check the accuracy against reference computations using `validate` notebooks and assemble an initial training set using `compile-dataset`.
