# Multifidelity Active Learning

An example that gradually runs computations in increasing over of complexity, 
revisiting whether the molecule is promising enough at each step.
The goal is to find a molecule with a large oxidation potential.

- _Simulation_ tasks are all using PM7 from MOPAC. The fidelity steps start from vertical ionization energies in vacuum,
   then vertical energies in solution, then finish with adiabatic in solution.
- _Machine Learning_ models are Gaussian Process Regression using the feature set from 
   [Doan et al.](Doan et al. <https://pubs.acs.org/doi/10.1021/acs.chemmater.0c00768>).
   The models predict the oxidation potential at the lowest fidelity and then the differences
   between each subsequent steps. We use known values for each level in place of the machine
   learning when available.
- _Active learning_ is based on expected improvement (EI). The next calculation to start after one finishes
  determined by first picking a level of fidelity randomly and then finding the calculation with the highest 
  EI that is ready to run that step.


   
## Running the example

Navigate to this directory and then call

```
examol run spec.py:spec
```

It will output to `run` and eventually produce a file, `report.md`, that contains a summary of run.

> Note: You will need to install Redis to run this example or change `num_workers` in `spec.py` to 1 then
> `colemna_queues` from `RedisQueues` to `colmena.queues.python.PipeQueues`. We use Redis to cope with the larger
> number of inference tasks produced by this application.
