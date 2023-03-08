"""Interface which launches an ExaMol run from the command line"""

from argparse import ArgumentParser

from examol import __version__


def main(args: list[str] | None = None):
    """Main function for the CLI"""

    # Make the parser and parse
    parser = ArgumentParser()

    parser.add_argument('--version', action='store_true', help='Print the ExaMol version and return')

    args = parser.parse_args(args)

    # Print the version
    if args.version:
        print(f'Running ExaMol version: {__version__}')
