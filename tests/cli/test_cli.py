import logging
from pathlib import Path

from examol.cli import main
from examol import __version__

_spec_dir = (Path(__file__).parent / '../../examples/redoxmers/').resolve()


def test_version(capsys):
    main(['--version'])
    cap = capsys.readouterr()
    assert __version__ in cap.out


def test_dryrun(caplog, capsys):
    with caplog.at_level(logging.INFO):
        main(['run', '--dry-run', f'{_spec_dir / "spec.py"}:spec'])
    assert 'dry run' in caplog.messages[-1]
