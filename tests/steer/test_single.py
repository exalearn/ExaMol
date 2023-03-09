"""Test single objective optimizer"""
import json
import logging

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from colmena.task_server import ParslTaskServer
from colmena.queue import ColmenaQueues, PipeQueues
from pytest import fixture, mark

from examol.score.rdkit import RDKitScorer, make_knn_model
from examol.select.baseline import RandomSelector
from examol.simulate.ase import ASESimulator
from examol.steer.single import SingleObjectiveThinker
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy


@fixture()
def recipe() -> RedoxEnergy:
    return RedoxEnergy(charge=1, energy_config='xtb', vertical=False)


@fixture()
def training_set(recipe) -> list[MoleculeRecord]:
    """Make a starting training set"""
    output = []
    for i, smiles in enumerate(['C', 'CC', 'CCC']):
        record = MoleculeRecord.from_identifier(smiles)
        record.properties[recipe.name] = {recipe.level: i}
        output.append(record)
    return output


@fixture()
def search_space() -> list[MoleculeRecord]:
    return [MoleculeRecord.from_identifier(x) for x in ['CO', 'CCCC', 'CCO']]


@fixture()
def scorer(recipe) -> RDKitScorer:
    pipeline = make_knn_model()
    return RDKitScorer(recipe.name, recipe.level, pipeline=pipeline)


@fixture()
def simulator(tmp_path) -> ASESimulator:
    return ASESimulator(scratch_dir=tmp_path)


@fixture()
def queues(recipe, scorer, simulator, tmp_path) -> ColmenaQueues:
    """Make a start the task server"""

    queues = PipeQueues(topics=['inference', 'simulation', 'train'])

    # Make parsl configuration
    config = Config(
        run_dir=str(tmp_path),
        executors=[HighThroughputExecutor(start_method='thread', max_workers=1)]
    )

    doer = ParslTaskServer(
        queues=queues,
        methods=[scorer.score, simulator.optimize_structure, scorer.retrain],
        config=config,
        timeout=15,
    )
    doer.start()

    yield queues
    queues.send_kill_signal()
    doer.join()


@fixture()
def thinker(queues, recipe, search_space, scorer, training_set, tmp_path) -> SingleObjectiveThinker:
    run_dir = tmp_path / 'run'
    return SingleObjectiveThinker(
        queues=queues,
        run_dir=run_dir,
        recipe=recipe,
        database=training_set,
        models=[scorer],
        selector=RandomSelector(10),
        num_workers=1,
        num_to_run=2,
        search_space=zip([x.identifier.smiles for x in search_space], scorer.transform_inputs(search_space)),
    )


@mark.timeout(10)
def test_thinker(thinker: SingleObjectiveThinker, training_set, caplog):
    caplog.set_level(logging.ERROR)

    # Make sure it is set up right
    assert len(thinker.search_space_keys) == 1
    assert len(thinker.database) == len(training_set)

    # Run it
    thinker.run()
    assert len(caplog.records) == 0, caplog.records[0]

    # Check the output files
    with (thinker.run_dir / 'inference-results.json').open() as fp:
        record = json.loads(fp.readline())
        assert record['success']

    assert len(thinker.database) >= len(training_set) + thinker.num_to_run


@mark.timeout(10)
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
