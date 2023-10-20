"""Tests for the specification class"""
from random import random
from pathlib import Path

from parsl import Config, HighThroughputExecutor
from proxystore.store import Store
from proxystore.connectors.file import FileConnector
from pytest import fixture
from sklearn.pipeline import Pipeline

from examol.score.base import Scorer
from examol.score.rdkit import RDKitScorer, make_knn_model
from examol.select.baseline import RandomSelector
from examol.simulate.ase import ASESimulator
from examol.specify import ExaMolSpecification
from examol.solution import SingleFidelityActiveLearning
from examol.steer.single import SingleStepThinker
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
    return RedoxEnergy(1, 'mopac_pm7')


@fixture()
def scorer() -> tuple[Scorer, Pipeline]:
    pipeline = make_knn_model()
    return RDKitScorer(), pipeline


@fixture()
def search_space(scorer, tmp_path) -> Path:
    scorer, _ = scorer  # Unpack

    molecules = ['C', 'CO', 'CN', 'CCl']
    # Store one copy as a SMILES file
    smi_path = Path(tmp_path) / 'search.smi'
    with smi_path.open('w') as fp:
        for mol in molecules:
            print(mol, file=fp)

    return smi_path


@fixture()
def selector() -> RandomSelector:
    return RandomSelector(2)


@fixture()
def simulator(tmp_path):
    return ASESimulator(scratch_dir=tmp_path)


@fixture()
def config(request, tmp_path) -> Config:
    """A basic, single configuration"""
    return Config(
        run_dir=str(tmp_path),
        executors=[
            HighThroughputExecutor(max_workers=1, address='localhost'),
        ]
    )


@fixture()
def spec(config, database, recipe, scorer, search_space, selector, simulator, tmp_path) -> ExaMolSpecification:
    scorer, pipeline = scorer
    solution = SingleFidelityActiveLearning(
        selector=selector,
        scorer=scorer,
        models=[[pipeline]],
        num_to_run=2,
    )
    return ExaMolSpecification(
        database=database,
        search_space=[search_space],
        solution=solution,
        simulator=simulator,
        recipes=[recipe],
        thinker=SingleStepThinker,
        compute_config=config,
        run_dir=tmp_path
    )


def test_database_load(spec):
    database = spec.load_database()
    assert len(database) == 3
    assert database[0].identifier.smiles == 'CC'


def test_assemble(spec):
    doer, thinker = spec.assemble()
    assert 'train' in doer.queues.topics


def test_split_config(spec):
    spec.compute_config = Config(
        executors=[
            HighThroughputExecutor(label='learning', max_workers=1, address='localhost'),
            HighThroughputExecutor(label='simulation', max_workers=1, address='localhost')
        ],
        run_dir=spec.compute_config.run_dir
    )
    spec.assemble()  # Should not error


def test_cache(spec):
    spec.assemble()
    assert Path(spec.run_dir / 'search-space').is_dir()
    last_config = Path(spec.run_dir / 'search-space' / 'settings.json').read_text()

    spec.assemble()  # Should not rebuild
    new_config = Path(spec.run_dir / 'search-space' / 'settings.json').read_text()
    assert last_config == new_config

    spec.thinker_options = {'inference_chunk_size': 2}  # Will force a rebuild
    spec.assemble()
    new_config = Path(spec.run_dir / 'search-space' / 'settings.json').read_text()
    assert last_config != new_config


def test_proxy(spec, tmpdir):
    file_store = Store(name='file', connector=FileConnector(tmpdir), metrics=True)

    # Test without any store
    doer, thinker = spec.assemble()
    assert all(x is None for x in thinker.queues.proxystore_name.values())

    # Test with a single store for everything
    spec.proxystore = file_store
    doer, thinker = spec.assemble()
    assert all(x == 'file' for x in thinker.queues.proxystore_name.values())

    # Only use file for the inference
    spec.proxystore = {'inference': file_store}
    doer, thinker = spec.assemble()
    assert all(x is None if n != "inference" else x == "file" for n, x in thinker.queues.proxystore_name.items()), thinker.queues.proxystore_name
