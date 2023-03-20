"""Reporting functions which write status to a markdown file in the run directory"""
import json
from datetime import datetime
import logging

import pandas as pd
from colmena.models import Result

from examol.reporting.base import BaseReporter
from examol.steer.base import MoleculeThinker

logger = logging.getLogger(__name__)


class MarkdownReporter(BaseReporter):
    """Write status of runs to a markdown file"""

    def report(self, thinker: MoleculeThinker):
        # Open the reporting file
        with (thinker.run_dir / 'report.md').open('w') as fo:
            print('# Run Report', file=fo)
            print(f'Report time: {datetime.now()}', file=fo)

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
                        node_hours += result['time_running'] / 3600
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
