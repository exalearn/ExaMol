# ExaMol
[![CI](https://github.com/exalearn/ExaMol/actions/workflows/python-app.yml/badge.svg)](https://github.com/exalearn/ExaMol/actions/workflows/python-app.yml)
[![Coverage Status](https://coveralls.io/repos/github/exalearn/ExaMol/badge.svg?branch=main)](https://coveralls.io/github/exalearn/ExaMol?branch=main)

Designing new molecules as fast as possible with AI and simulation

## Installation

You can build the entire environment with Anaconda:

```commandline
conda env create --file envs/environment-cpu.yml --force
```

The above command builds the environment for a commodity CPU. 
We will provide environments compatible with supercomputers over time.

## Using ExaMol

ExaMol deploys a computational workflow following a specification which that contains information including:
- what is molecular properties are already known,
- what are the molecules we could search,
- the types of computations being performed
- how to search over them (e.g., how to schedule tasks, which active learning strategy, informed by which ML model),
- and the resources over which computations are deployed.

An example which performs a random search using xTB would look something like

```python
spec = ExaMolSpecification(
    database='training-data.json',
    recipe=RedoxEnergy(charge=1, compute_config='xtb'),
    search_space='search_space.smi',
    selector=RandomSelector(8),
    simulator=ASESimulator(scratch_dir='/tmp'),
    scorer=RDKitScorer(pipeline=KNeighborsRegressor()),
    num_to_run=8,
    thinker=SingleObjectiveThinker,
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
