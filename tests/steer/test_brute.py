"""Test brute force search"""
import logging

from pytest import fixture

from examol.specify import SolutionSpecification
from examol.start.fast import RandomStarter
from examol.steer.baseline import BruteForceThinker


@fixture()
def thinker(queues, recipe, search_space, database, tmp_path) -> BruteForceThinker:
    run_dir = tmp_path / 'run'
    solution = SolutionSpecification(
        starter=RandomStarter(),
        num_to_run=3,
    )
    return BruteForceThinker(
        queues=queues,
        run_dir=run_dir,
        recipes=[recipe],
        database=database,
        num_workers=1,
        solution=solution,
        search_space=[search_space],
    )


def test_thinker(thinker, caplog, database):
    caplog.set_level(logging.ERROR)

    # Run it
    start_size = len(database)
    thinker.run()
    assert len(caplog.records) == 0, caplog.records[0]

    # Make sure it ran the target number of molecules
    assert len(thinker.database) >= start_size + thinker.num_to_run
