"""Test single objective optimizer"""
import json
import logging
from pathlib import Path

from pytest import fixture, mark

from examol.select.baseline import RandomSelector
from examol.simulate.base import SimResult
from examol.simulate.initialize import add_initial_conformer
from examol.solution import SingleFidelityActiveLearning
from examol.start.fast import RandomStarter
from examol.steer.single import SingleStepThinker
from examol.store.models import Conformer
from examol.store.recipes import SolvationEnergy


@fixture()
def thinker(queues, recipe, search_space, scorer, database, tmpdir, pool) -> SingleStepThinker:
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
        pool=pool,
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
    assert any('Too few to entries to train oxidation_potential. Waiting for 4' in x for x in run_log)

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
    assert 'HKRAFRNOHKOEOU-UHFFFAOYSA-N' in caplog.messages[-1]  # That's the InChI key for C1C2...


@mark.timeout(120)
def test_multiproperty_iterator(thinker, caplog, recipe):
    # Add solvation energy as a record and a record for 'C' which has a nuetral geometry
    thinker.recipes = [SolvationEnergy(recipe.energy_config, solvent='acn'), recipe]

    record = thinker.database.get_or_make_record('C')
    record = add_initial_conformer(record)
    new_conf = Conformer.from_simulation_result(
        SimResult(xyz=record.conformers[0].xyz, config_name=recipe.energy_config, charge=0, energy=1, solvent=None)
    )
    record.conformers.append(new_conf)
    thinker.database.update_record(record)

    # Insert a molecule and make sure it yields three computations
    thinker.task_queue.append(('C', 0))

    requests = [next(thinker.task_iterator), next(thinker.task_iterator)]
    assert requests[0][0].identifier.smiles == 'C'  # It should not
    assert 1 in [r[-1].charge for r in requests]
