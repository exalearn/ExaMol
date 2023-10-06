"""Tests for combinations of recipe and molecule which fail"""
from shutil import rmtree
from sys import platform

from pytest import fixture, mark

from examol.simulate.ase import ASESimulator
from examol.store.models import MoleculeRecord
from examol.store.recipes.redox import RedoxEnergy

is_mac = platform.startswith("darwin")


@fixture()
def simulator() -> ASESimulator:
    simulator = ASESimulator(clean_after_run=False)
    rmtree(simulator.scratch_dir, ignore_errors=True)
    return simulator


@mark.skipif(is_mac, reason='No xTB on OSX tests')
@mark.parametrize('smiles,charge', [
    ('N#CC1OC2CC2C1=O', 1),
])
def test_no_relaxed_charged(smiles: str, charge: int, simulator: ASESimulator):
    """A test where we do not create a new conformer after relaxation"""

    # Make the problem case
    record = MoleculeRecord.from_identifier(smiles)
    recipes = [
        RedoxEnergy(energy_config='xtb', charge=charge, vertical=True),
        RedoxEnergy(energy_config='xtb', charge=charge, vertical=False),
        RedoxEnergy(energy_config='mopac_pm7', charge=charge, vertical=False),
        RedoxEnergy(energy_config='mopac_pm7', charge=charge, vertical=True),
    ]

    # Perform the recipe
    for recipe in recipes:
        while len(suggestions := recipe.suggest_computations(record)) > 0:
            for suggestion in suggestions:
                if suggestion.optimize:
                    result, steps, _ = simulator.optimize_structure(record.key, suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent)
                    assert record.add_energies(result, steps), 'No new conformer was added'
                else:
                    result, _ = simulator.compute_energy(record.key, suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent)
                    record.add_energies(result)

        # Compute the result
        try:
            recipe.update_record(record)
        except ValueError:
            with open('failed.json', 'w') as fp:
                print(record.to_json(indent=2), file=fp)
            raise
