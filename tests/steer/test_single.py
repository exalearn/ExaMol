"""Test single objective optimizer"""
import json
import logging
from pathlib import Path

from pytest import fixture, mark

from examol.select.baseline import RandomSelector
from examol.solution import SingleFidelityActiveLearning
from examol.start.fast import RandomStarter
from examol.steer.single import SingleStepThinker


@fixture()
def thinker(queues, recipe, search_space, scorer, database, tmpdir) -> SingleStepThinker:
    run_dir = Path(tmpdir / 'run')
    scorer, model = scorer
    solution = SingleFidelityActiveLearning(
        scorer=scorer,
        starter=RandomStarter(),
        models=[[model, model]],
        selector=RandomSelector(10),
        minimum_training_size=4,
        num_to_run=3,
    )
    return SingleStepThinker(
        queues=queues,
        run_dir=run_dir,
        recipes=[recipe],
        database=database,
        num_workers=1,
        solution=solution,
        search_space=[search_space],
    )


@mark.timeout(120)
def test_thinker(thinker: SingleStepThinker, database, caplog):
    caplog.set_level(logging.ERROR)

    # Make sure it is set up right
    assert len(thinker.search_space_smiles) == 1
    start_size = len(thinker.database)

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
    assert record_count >= thinker.num_to_run

    assert len(thinker.database) >= start_size + thinker.num_to_run


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
