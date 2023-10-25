# ExaMol
[![CI](https://github.com/exalearn/ExaMol/actions/workflows/python-app.yml/badge.svg)](https://github.com/exalearn/ExaMol/actions/workflows/python-app.yml)
[![Deploy Docs](https://github.com/exalearn/ExaMol/actions/workflows/gh-pages.yml/badge.svg)](https://exalearn.github.io/ExaMol/)
[![Coverage Status](https://coveralls.io/repos/github/exalearn/ExaMol/badge.svg?branch=main)](https://coveralls.io/github/exalearn/ExaMol?branch=main)

Designing new molecules as fast as possible with AI and simulation.

- Documentation: [exalearn.github.io/ExaMol/](https://exalearn.github.io/ExaMol/)
- Source Code: [github.com/exalearn/ExaMol](https://github.com/exalearn/ExaMol)

## Installation

First clone this repository to the computer you'll use to design molecules:

```commandline
git clone https://github.com/exalearn/ExaMol.git
```

You can then build the entire environment with Anaconda:

```commandline
cd ExaMol
conda env create --file envs/environment-cpu.yml --force
```

The above command builds the environment for a commodity CPU. 
We will provide environments compatible with supercomputers over time.

## Using ExaMol

ExaMol deploys a computational workflow following a specification which that contains information including:
- what is molecular properties are already known,
- what are the molecules we could search,
- the types of computations being performed,
- how to search over them (e.g., how to schedule tasks, which active learning strategy, informed by which ML model),
- and the resources over which computations are deployed.

An example which performs a greedy search using xTB would look something like

```python
recipe = RedoxEnergy(charge=1, compute_config='xtb')  # What we're trying to optimize
solution = SingleFidelityActiveLearning(  # How we are going to optimize it
    starter=RandomStarter(),
    minimum_training_size=4,
    scorer=RDKitScorer(),
    models=[[KNeighborsRegressor()]],
    selector=GreedySelector(10, maximize=True),
    num_to_run=8,
)
spec = ExaMolSpecification(  # How to set up ExaMol
    database=(my_path / 'training-data.json'),
    recipes=[recipe],
    search_space=[(my_path / 'search_space.smi')],
    solution=solution,
    simulator=ASESimulator(scratch_dir='./tmp'),
    thinker=SingleStepThinker,
    thinker_options=dict(num_workers=2),
    compute_config=config,
    run_dir='run'
)
```

The full example is shown in our [examples directory](./examples)

### Running an Example

The redoxmer example we allude to above can be run from the command line and requires only a few minutes to complete:

```commandline
examol run examples/redoxmers/spec.py:spec
```

The above command simply tells ExaMol the name of the file containing the run spec, and the name of the spec within that file.
ExaMol will then run it for you.

## Project Support

Initial development of ExaMol was funded jointly by 
the ExaLearn Co-design Center of the Department of Energy Exascale Computing Project 
and the Joint Center for Energy Storage Research (JCESR), an Energy Innovation Hub funded by the Department of Energy.
