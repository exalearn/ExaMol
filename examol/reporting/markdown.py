"""Reporting functions which write status to a markdown file in the run directory"""
import json
import logging
from typing import TextIO
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pylab as plt

from examol.reporting.base import BaseReporter
from examol.steer.base import MoleculeThinker
from examol.steer.single import SingleObjectiveThinker

logger = logging.getLogger(__name__)


class MarkdownReporter(BaseReporter):
    """Write status of runs to a markdown file"""

    def report(self, thinker: MoleculeThinker):
        logger.info(f'Starting to process results in: {thinker.run_dir}')
        # Open the reporting file
        with (thinker.run_dir / 'report.md').open('w') as fo:
            print('# Run Report', file=fo)
            print(f'Report time: {datetime.now()}', file=fo)

            # Print out the different types of data
            self._write_task_summary(fo, thinker)
            if isinstance(thinker, SingleObjectiveThinker):
                self._plot_over_time(fo, thinker)

    def _write_task_summary(self, fo: TextIO, thinker: MoleculeThinker):
        """Summarize the tasks running on the summary

        Args:
            fo: File to write to
            thinker: Thinker being assessed
        """
        # Count how many jobs of each type have run
        task_summary = []
        result_files = thinker.run_dir.glob('*-results.json')
        for result_file in result_files:
            task_type = result_file.name[:-len('-results.json')]  # Strip off the suffix
            count = node_hours = failures = 0
            with result_file.open() as fp:
                for line in fp:
                    result = json.loads(line)
                    count += 1
                    node_hours += (result['time_running'] or 0) / 3600
                    failures += not result['success']
            task_summary.append({
                'Task Type': task_type,
                'Count': count,
                'Node Hours': f'{node_hours:.2g}',
                'Failures': f'{failures} ({failures / count * 100.:.1f}%)'
            })

        # Save run summary to output file
        task_summary = pd.DataFrame(task_summary)
        print('\n## Task Summary\nMeasures how many tasks have run as part of the application', file=fo)
        print('\n' + task_summary.to_markdown(index=False, tablefmt='github'), file=fo)

    def _plot_over_time(self, fo: TextIO, thinker: SingleObjectiveThinker):
        """Plot the properties evaluated over time"""

        # Exit if simulation results do not exist yet
        simulation_results = thinker.run_dir / 'simulation-results.json'
        if not simulation_results.exists():
            return

        # Load in the simulation results. The computed property is stored in 'value'
        results = []
        with simulation_results.open() as fp:
            for line in fp:
                record = json.loads(line)
                if 'result' in record['task_info']:
                    results.append(record['task_info']['result'])
        logger.info(f'Found {len(results)} molecule records')

        # Make a figure
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.scatter(
            np.arange(len(results)),
            results,
        )
        ax.set_xlabel('Result')
        ax.set_ylabel(f'{thinker.recipes.name}@\n{thinker.recipes.level}')

        fig.tight_layout()
        fig.savefig(thinker.run_dir / 'simulation-outputs.png', dpi=320)

        # Write the markdown
        print('\n## Outcomes over Time\nThe property of the molecules over time.', file=fo)
        print('\n![simulation](simulation-outputs.png)', file=fo)
