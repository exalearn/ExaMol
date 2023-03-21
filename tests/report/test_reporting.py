import shutil
from pathlib import Path

from pytest import fixture

from examol.reporting.markdown import MarkdownReporter
from examol.steer.single import SingleObjectiveThinker
from examol.store.recipes import RedoxEnergy

example_dir = Path(__file__).parent / 'example'


class FakeThinker(SingleObjectiveThinker):
    run_dir = example_dir
    recipe = RedoxEnergy(charge=1, energy_config='xtb')

    def __init__(self):
        pass


@fixture()
def thinker():
    return FakeThinker()


def test_markdown(thinker, tmpdir):
    run_dir = Path(tmpdir) / 'test-dir'
    shutil.copytree(example_dir, run_dir)
    thinker.run_dir = run_dir
    reporter = MarkdownReporter()
    reporter.report(thinker)
    assert (thinker.run_dir / 'report.md').is_file()
    assert (thinker.run_dir / 'simulation-outputs.png').is_file()
