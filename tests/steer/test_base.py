"""Tests for the base thinker"""
import gzip

from pytest import mark
from colmena.thinker.resources import ResourceCounter

from examol.store.models import MoleculeRecord
from examol.specify import SolutionSpecification
from examol.steer.base import MoleculeThinker


@mark.parametrize('use_json', [True, False])
def test_search_space(queues, search_space, tmp_path, database, use_json, pool):
    """Test using a JSON-format search space"""

    # Save the training data to JSON format
    json_search_space = search_space.parent / 'search_space.json.gz'
    with search_space.open() as fi, gzip.open(json_search_space, 'wt') as fo:
        for mol in fi:
            record = MoleculeRecord.from_identifier(mol.strip())
            print(record.json(), file=fo)

    # Make the solution specification
    solution = SolutionSpecification(
        num_to_run=3,
    )

    thinker = MoleculeThinker(
        queues=queues,
        rec=ResourceCounter(8),
        recipes=(),
        solution=solution,
        run_dir=tmp_path / 'run',
        search_space=[json_search_space] if use_json else [search_space],
        database=database,
        pool=pool
    )
    assert len(list(thinker.iterate_over_search_space())) == 5
    smiles_only = list(thinker.iterate_over_search_space(only_smiles=True))
    assert len(smiles_only) == 5
    assert smiles_only[-1] == record.identifier.smiles
