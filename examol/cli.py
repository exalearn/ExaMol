"""Interface which launches an ExaMol run from the command line"""
import logging
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import sleep

from rdkit import RDLogger
from proxystore.store import get_store

from examol import __version__
from examol.specify import ExaMolSpecification

logger = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*')


def load_spec(path: Path, var_name: str) -> ExaMolSpecification:
    """Load the specification from a file

    Args:
        path: Path to the file contained the specification
        var_name: Name of the specification variable within this file
    Returns:
        Specification
    """
    assert path.is_file(), f'No such file: {path}'
    path = path.absolute()

    old_path = Path().cwd()  # Store where we initial are
    try:
        os.chdir(path.parent)  # Go to the directory with the specification file
        spec_ns = {}
        exec(path.read_text(), spec_ns)
    finally:
        os.chdir(old_path)  # Make sure to change back
    assert var_name in spec_ns, f'Variable {var_name} not found in {path}'

    return spec_ns[var_name]


def run_examol(args):
    """Run ExaMol"""

    # Load configuration
    assert ":" in args.spec, "Specification must be defined as <file_path>:<variable_name>. Call `examol run -h`"
    spec_path, spec_name = args.spec.split(":")
    spec_path = Path(spec_path)
    spec = load_spec(spec_path, spec_name)
    logger.info(f'Loaded specification from {spec_path}, where it was named {spec_name}')

    # Create the thinker and doer, then print out key statistics
    with spec.assemble() as (doer, thinker, store):
        logger.info(f'Created a task server with methods: {doer.method_names}')
        logger.info(f'Created a thinker based on {thinker.__class__}')
        logger.info(f'Will run a total of {spec.solution.num_to_run} molecules')
        logger.info(f'Will save results into {thinker.run_dir}')

        # Make a function to clear the proxystore caches
        def _clear_stores():
            names = set(thinker.queues.proxystore_name.values())
            logger.info(f'There are {len(names)} proxystore caches to clear.')
            for name in names:
                if name is not None:
                    logger.info(f'Closing proxystore: {name}')
                    store = get_store(name)
                    store.close()

        # Stop if we are in a dry run
        if args.dry_run:
            _clear_stores()
            logger.info('Finished with dry run. Exiting')
            return

        # Launch them
        doer.start()  # Run the doer as a subprocess
        reporter_threads = []
        try:
            thinker.start()  # Start the thinker
            logger.info('Launched the thinker. Waiting a second before launching the reporters')

            # Start the monitors
            sleep(1.)
            for reporter in spec.reporters:
                reporter_threads.append(reporter.monitor(thinker, args.report_freq))
            logger.info('Launched the reporting threads')

            # Make sure the doer is alive
            if not doer.is_alive():  # no-coverage:
                doer.join()
                raise ValueError(f'Doer process exited with status code {doer.exitcode}')
            thinker.join(timeout=args.timeout)  # Wait until it completes
        except TimeoutError:
            logger.info('Hit timeout. Sending stop signal to Thinker then blocking until all ongoing tasks complete')
            thinker.done.set()  # If it hits the timeout
            thinker.join()
        finally:
            logger.info('Thinker complete, sending a signal to shut down the doer')
            thinker.done.set()  # Make sure it is set
            thinker.queues.send_kill_signal()
            doer.join()
    logger.info('All processes have completed.')

    # Once complete, run the reporting one last time
    for reporter, thread in zip(spec.reporters, reporter_threads):
        logger.info(f'Waiting for {reporter} thread to complete.')
        thread.join()

        logger.info(f'Running {reporter} a last time.')
        reporter.report(thinker)

    logger.info(f'Find run details in {spec.run_dir.absolute()}')

    # Clear out the proxystore cache
    _clear_stores()


def main(args: list[str] | None = None):
    """Main function for the CLI"""

    # Make the parser and parse
    parser = ArgumentParser()
    parser.add_argument('--version', action='store_true', help='Print the ExaMol version and return')

    subparsers = parser.add_subparsers(dest='command')

    subparser = subparsers.add_parser('run', help='Run ExaMol')
    subparser.add_argument('--dry-run', action='store_true', help='Load in configuration but do not start computing')
    subparser.add_argument('--report-freq', default=600, type=float, help='How often to write run status (units: s)')
    subparser.add_argument('--timeout', default=None, type=float, help='Maximum time to let ExaMol run (units: s)')
    subparser.add_argument('spec', help='Path to the run specification. Format is the path to a Python file containing the spec, '
                                        'followed by a colon and the name of the variable defining the specification (e.g., `spec.py:spec`)')

    args = parser.parse_args(args)

    # Print the version
    if args.version:
        print(f'Running ExaMol version: {__version__}')
        return

    # Turn on logging
    examol_logger = logging.getLogger('examol')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    examol_logger.addHandler(handler)
    examol_logger.setLevel(logging.INFO)
    examol_logger.info(f'Starting ExaMol v{__version__}')

    # Run ExaMol
    if args.command == 'run':
        run_examol(args)
    else:
        raise NotImplementedError()
