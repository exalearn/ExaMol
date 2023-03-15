"""Tests for the specification class"""
from random import random
from pathlib import Path
from csv import writer
import json

from parsl import Config
from parsl.configs import htex_local
from pytest import fixture, raises
from sklearn.pipeline import Pipeline

from examol.score.base import Scorer
from examol.score.rdkit import RDKitScorer, make_knn_model
from examol.select.baseline import RandomSelector
from examol.simulate.ase import ASESimulator
from examol.specify import ExaMolSpecification
from examol.steer.single import SingleObjectiveThinker
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy


@fixture()
def database(recipe, tmp_path) -> Path:
    path = Path(tmp_path) / 'test.records'
    with path.open('w') as fp:
        for mol in ['CC', 'O', 'N']:
            record = MoleculeRecord.from_identifier(mol)
            record.properties[recipe.name] = {recipe.level: random()}
            print(record.to_json(), file=fp)
    return path


@fixture()
def recipe() -> RedoxEnergy:
    return RedoxEnergy(1, 'xtb')


@fixture()
def scorer(recipe) -> tuple[Scorer, Pipeline]:
    pipeline = make_knn_model()
    return RDKitScorer(recipe), pipeline


@fixture()
def search_space(scorer, tmp_path) -> tuple[Path, Path]:
    scorer, _ = scorer  # Unpack

    molecules = ['C', 'CO', 'CN', 'CCl']
    # Store one copy as a SMILES file
    smi_path = Path(tmp_path) / 'search.smi'
    with smi_path.open('w') as fp:
        for mol in molecules:
            print(mol, file=fp)

    # Store another that's pre-processed
    csv_path = Path(tmp_path) / 'search.csv'
    records = [MoleculeRecord.from_identifier(s) for s in molecules]
    inputs = [json.dumps(x) for x in scorer.transform_inputs(records)]
    with csv_path.open('w') as fp:
        csv = writer(fp)
        for row in zip(molecules, inputs):
            csv.writerow(row)
    return smi_path, csv_path


@fixture()
def selector() -> RandomSelector:
    return RandomSelector(2)


@fixture()
def simulator(tmp_path):
    return ASESimulator(scratch_dir=tmp_path)


@fixture()
def config(tmp_path) -> Config:
    config = htex_local.config
    config.run_dir = tmp_path
    return config


@fixture()
def spec(config, database, recipe, scorer, search_space, selector, simulator, tmp_path) -> ExaMolSpecification:
    scorer, pipeline = scorer
    return ExaMolSpecification(
        database=database,
        search_space=search_space[0],
        selector=selector,
        scorer=scorer,
        models=[pipeline],
        simulator=simulator,
        recipe=recipe,
        thinker=SingleObjectiveThinker,
        compute_config=config,
        num_to_run=2,
        run_dir=tmp_path
    )


def test_database_load(spec):
    database = spec.load_database()
    assert len(database) == 3
    assert database[0].identifier.smiles == 'CC'


def test_search_load(search_space, spec):
    for path in search_space:
        spec.search_space = path
        space = list(spec.load_search_space())
        assert len(space) == 4
        assert space[0] == ('C', 'C')

    with raises(ValueError):
        spec.search_space = 'not.supported'
        next(spec.load_search_space())


def test_assemble(spec):
    doer, thinker = spec.assemble()
    assert 'train' in doer.queues.topics
