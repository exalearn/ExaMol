import sys
import logging
from pathlib import Path

from pytest import mark

from examol.cli import main
from examol import __version__

_spec_dir = (Path(__file__).parent / '../../examples/redoxmers/').resolve()
on_mac = (sys.platform == 'darwin')


def test_version(capsys):
    main(['--version'])
    cap = capsys.readouterr()
    assert __version__ in cap.out


def test_dryrun(caplog, capsys):
    with caplog.at_level(logging.INFO):
        main(['run', '--dry-run', f'{_spec_dir / "spec.py"}:spec'])
    assert 'dry run' in caplog.messages[-1]


@mark.skipif(on_mac, reason='Only test the CLI on Linux')
@mark.timeout(240)
def test_full(caplog):
    with caplog.at_level(logging.INFO):
        main(['run', '--report-freq', '10', f'{_spec_dir / "spec.py"}:spec'])
    assert any('Find run details in' in m for m in caplog.messages[-4:])

    # Make sure the database got reported
    assert (_spec_dir / 'run' / 'database.json').is_file()


@mark.skipif(on_mac, reason='Only test the CLI on Linux')
@mark.timeout(240)
def test_timeout(caplog):
    with caplog.at_level(logging.INFO):
        main(['run', '--timeout', '0', '--report-freq', '10', f'{_spec_dir / "spec.py"}:spec'])
    assert any(m.startswith('Find run details in') for m in caplog.messages)

    # Make sure the database got reported
    assert (_spec_dir / 'run' / 'database.json').is_file()
