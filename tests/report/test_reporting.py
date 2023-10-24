import shutil
from time import sleep
from pathlib import Path
from threading import Event

from colmena.models import Result
from pytest import fixture, mark

from examol.reporting.markdown import MarkdownReporter
from examol.steer.single import SingleStepThinker
from examol.store.recipes import RedoxEnergy

example_dir = Path(__file__).parent / 'example'


class FakeThinker(SingleStepThinker):
    run_dir = example_dir
    recipes = [RedoxEnergy(charge=1, energy_config='xtb')]

    def __init__(self):
        pass

    def _write_result(self, result: Result, result_type: str):
        with (self.run_dir / f'{result_type}-results.json').open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)


@fixture()
def thinker(tmpdir):
    thinker = FakeThinker()
    run_dir = Path(tmpdir) / 'test-dir'
    shutil.copytree(example_dir, run_dir)
    thinker.run_dir = run_dir
    thinker.done = Event()
    return thinker


def test_markdown(thinker):
    reporter = MarkdownReporter()
    reporter.report(thinker)
    assert (thinker.run_dir / 'report.md').is_file()
    assert (thinker.run_dir / 'simulation-outputs_recipe-0.png').is_file()


@mark.timeout(25)
def test_monitor(thinker):
    reporter = MarkdownReporter()
    thread = reporter.monitor(thinker, frequency=1)
    thread.join(1)
    assert not thinker.done.is_set()
    assert thread.is_alive()

    # Make sure it writes after at least 2s
    sleep(2)
    assert thread.is_alive(), thread.join()
    assert (thinker.run_dir / 'report.md').is_file()

    # Make sure it shuts down
    thinker.done.set()
    sleep(2)
    assert not thread.is_alive()
