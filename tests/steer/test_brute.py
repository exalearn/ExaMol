"""Test brute force search"""
import logging

from pytest import fixture

from examol.specify import SolutionSpecification
from examol.start.fast import RandomStarter
from examol.steer.baseline import BruteForceThinker


@fixture()
def thinker(queues, recipe, search_space, training_set, tmp_path) -> BruteForceThinker:
    run_dir = tmp_path / 'run'
    solution = SolutionSpecification(
        starter=RandomStarter(),
        num_to_run=3,
    )
    return BruteForceThinker(
        queues=queues,
        run_dir=run_dir,
        recipes=[recipe],
        database=training_set,
        num_workers=1,
        solution=solution,
        search_space=[search_space],
    )


def test_thinker(thinker, caplog, training_set):
    caplog.set_level(logging.ERROR)

    # Run it
    thinker.run()
    assert len(caplog.records) == 0, caplog.records[0]

    # Make sure it ran the target number of molecules
    assert len(thinker.database) >= len(training_set) + thinker.num_to_run
