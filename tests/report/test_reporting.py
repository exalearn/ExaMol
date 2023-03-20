from pathlib import Path

from pytest import fixture

from examol.steer.base import MoleculeThinker
from examol.reporting.markdown import BaseReporter, MarkdownReporter

example_dir = Path(__file__).parent / 'example'


class FakeThinker(MoleculeThinker):
    run_dir = example_dir

    def __init__(self):
        pass


@fixture()
def thinker():
    return FakeThinker()


def test_markdown(thinker):
    reporter = MarkdownReporter()
    reporter.report(thinker)
