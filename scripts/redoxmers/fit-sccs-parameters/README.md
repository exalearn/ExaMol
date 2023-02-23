# Fitting SCCS Parameters

The self-consistent continuum solvation (SCCS) model used in CP2K does not come pre-packaged with parameters describing each solvent.
We are going to fit them by comparing them to the solvation energies available in [our previous work](https://pubs.acs.org/doi/abs/10.1021/acs.jpca.1c01960).

## Background

We are going to use the SCCS of [Andreussi et al. (2012)](https://aip.scitation.org/doi/figure/10.1063/1.3676407) which has 6 adjustable parameters:

- *&epsilon;<sub>0</sub>*: Dielectric constant, form experiment
- *&gamma;*: Surface energy, from experiment
- *&alpha;,&beta;*: solvent-specific parameters to the cavity, which must be fit
- *&rho;<sub>min</sub>,&rho;<sub>max</sub>*: Parameters which control the smoothness of the dielectric function. We'll use defaults

[The CP2K developers note](https://groups.google.com/g/cp2k/c/7oYTqSIyIqI/m/7D62tXIzBgAJ) the choice of &alpha; and &beta; should not matter, and the defaults for &rho; should be ok. 
We tested at least the &alpha; and &beta; before getting their advice. 