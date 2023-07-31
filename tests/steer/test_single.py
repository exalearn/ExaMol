"""Test single objective optimizer"""
import json
import logging
from pathlib import Path

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from colmena.task_server import ParslTaskServer
from colmena.queue import ColmenaQueues, PipeQueues
from proxystore.connectors.file import FileConnector
from proxystore.store import Store, register_store
from pytest import fixture, mark
from sklearn.pipeline import Pipeline

from examol.score.rdkit import RDKitScorer, make_knn_model
from examol.select.baseline import RandomSelector
from examol.simulate.ase import ASESimulator
from examol.start.fast import RandomStarter
from examol.steer.single import SingleStepThinker
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy


@fixture()
def recipe() -> RedoxEnergy:
    return RedoxEnergy(charge=1, energy_config='xtb', vertical=True)


@fixture()
def training_set(recipe) -> list[MoleculeRecord]:
    """Make a starting training set"""
    output = []
    for i, smiles in enumerate(['CCCC', 'CCO']):
        record = MoleculeRecord.from_identifier(smiles)
        record.properties[recipe.name] = {recipe.level: i}
        output.append(record)
    return output


@fixture()
def search_space(tmp_path) -> Path:
    path = tmp_path / 'search-space.smi'
    with path.open('w') as fp:
        for s in ['C', 'N', 'O', 'Cl', 'S']:
            print(s, file=fp)
    return path


@fixture()
def scorer() -> tuple[RDKitScorer, Pipeline]:
    pipeline = make_knn_model()
    return RDKitScorer(), pipeline


@fixture()
def simulator(tmp_path) -> ASESimulator:
    return ASESimulator(scratch_dir=tmp_path)


@fixture()
def queues(recipe, scorer, simulator, tmp_path) -> ColmenaQueues:
    """Make a start the task server"""
    # Unpack inputs
    scorer, _ = scorer

    # Make the queues
    store = Store(name='file', connector=FileConnector(store_dir=str(tmp_path)))
    register_store(store, exist_ok=True)
    queues = PipeQueues(topics=['inference', 'simulation', 'train'], proxystore_name=store.name)

    # Make parsl configuration
    config = Config(
        run_dir=str(tmp_path),
        executors=[HighThroughputExecutor(start_method='spawn', max_workers=1, address='localhost')]
    )

    doer = ParslTaskServer(
        queues=queues,
        methods=[scorer.score, simulator.optimize_structure, simulator.compute_energy, scorer.retrain],
        config=config,
        timeout=15,
    )
    doer.start()

    yield queues
    queues.send_kill_signal()
    doer.join()


@fixture()
def thinker(queues, recipe, search_space, scorer, training_set, tmp_path) -> SingleStepThinker:
    run_dir = tmp_path / 'run'
    scorer, model = scorer
    return SingleStepThinker(
        queues=queues,
        run_dir=run_dir,
        recipes=[recipe],
        database=training_set,
        scorer=scorer,
        starter=RandomStarter(4, 1),
        models=[[model, model]],
        selector=RandomSelector(10),
        num_workers=1,
        num_to_run=3,
        search_space=[search_space],
    )


@mark.timeout(120)
def test_thinker(thinker: SingleStepThinker, training_set, caplog):
    caplog.set_level(logging.ERROR)

    # Make sure it is set up right
    assert len(thinker.search_space_keys) == 1
    assert len(thinker.database) == len(training_set)

    # Run it
    thinker.run()
    assert len(caplog.records) == 0, caplog.records[0]

    # Check if there are points where the
    run_log = (thinker.run_dir / 'run.log').read_text().splitlines(keepends=False)
    assert any('Training set is smaller than the threshold size (2<4)' in x for x in run_log)
    assert any('Too few to entries to train. Waiting for 4' in x for x in run_log)

    # Check the output files
    with (thinker.run_dir / 'inference-results.json').open() as fp:
        record = json.loads(fp.readline())
        assert record['success']

    # Make sure we have more than a few simulation records
    with (thinker.run_dir / 'simulation-records.json').open() as fp:
        record_count = sum(1 for _ in fp)
    assert record_count > thinker.num_to_run

    assert len(thinker.database) >= len(training_set) + thinker.num_to_run


@mark.timeout(120)
def test_iterator(thinker, caplog):
    caplog.set_level('WARNING')

    # Insert a bad and good SMILES into the task queue
    thinker.task_queue.append(('C1C2CN3C1C1C3CN21', 1))  # XYZ generation fails
    thinker.task_queue.append(('C', 0))

    # Get the next task
    record = next(thinker.task_iterator)[0]
    assert record.identifier.smiles == 'C'  # It should not

    # Make sure we are warned about it
    assert 'C1C2CN3C1C1C3CN21' in caplog.messages[-1]
