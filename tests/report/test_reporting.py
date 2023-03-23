import shutil
from time import sleep
from pathlib import Path
from threading import Event

from pytest import fixture, mark

from examol.reporting.database import DatabaseWriter
from examol.reporting.markdown import MarkdownReporter
from examol.steer.single import SingleObjectiveThinker
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy

example_dir = Path(__file__).parent / 'example'


class FakeThinker(SingleObjectiveThinker):
    run_dir = example_dir
    recipe = RedoxEnergy(charge=1, energy_config='xtb')

    def __init__(self):
        pass


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
    assert (thinker.run_dir / 'simulation-outputs.png').is_file()


def test_database(thinker):
    record = MoleculeRecord.from_identifier('C')
    thinker.database = {record.key: record}
    reporter = DatabaseWriter()
    reporter.report(thinker)
    assert (thinker.run_dir / 'database.json').is_file()


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
