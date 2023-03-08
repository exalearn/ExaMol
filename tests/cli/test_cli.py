from examol.cli import main
from examol import __version__


def test_version(capsys):
    main(['--version'])
    cap = capsys.readouterr()
    assert __version__ in cap.out
