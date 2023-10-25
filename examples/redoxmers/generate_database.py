"""Generate the initial database and search space"""
from sklearn.model_selection import train_test_split
from examol.simulate.ase import ASESimulator
from examol.simulate.initialize import generate_inchi_and_xyz
from examol.store.models import MoleculeRecord
from examol.store.recipes.redox import RedoxEnergy
from rdkit import RDLogger
from foundry import Foundry
from pathlib import Path
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# First step is to generate the
client = Foundry()
client.load('10.18126/jos5-wj65', globus=False)
qm9 = client.load_data()['train'][0]
print(f'Downloaded QM9 dataset from Foundry. Total size: {len(qm9)}')

# Now sample a set of 16 molecules to start with
train_qm9, search_qm9 = train_test_split(qm9, train_size=17, test_size=1000, random_state=1)
print(f'Split off {len(train_qm9)} molecules to use as a training set')

# Compute initial xTB results for the training set, save them as JSON records
simulator = ASESimulator()
out_file = Path('training-data.json')
out_file.unlink(missing_ok=True)
recipe = RedoxEnergy(1, energy_config='xtb', solvent='acn')
for smiles in tqdm(train_qm9['smiles_0'], desc='xTB'):
    # Initialize the record
    record = MoleculeRecord.from_identifier(smiles)

    # Run the optimizations
    try:
        _, xyz = generate_inchi_and_xyz(smiles)
    except ValueError:
        continue
    neutral_result, _, _ = simulator.optimize_structure(xyz, config_name='xtb', charge=0)
    assert record.add_energies(neutral_result)
    charged_result, charged_steps, _ = simulator.optimize_structure(neutral_result.xyz, config_name='xtb', charge=1)
    assert record.add_energies(charged_result, charged_steps)

    # Do them in solvents
    neutral_solvent, _ = simulator.compute_energy(neutral_result.xyz, config_name='xtb', charge=0, solvent='acn')
    assert not record.add_energies(neutral_solvent)
    charged_solvent, _ = simulator.compute_energy(charged_result.xyz, config_name='xtb', charge=1, solvent='acn')
    assert not record.add_energies(charged_solvent)

    # Save them
    assert recipe.update_record(record) is not None
    with out_file.open('a') as fp:
        print(record.to_json(), file=fp)
print(f'Saved xTB results to {out_file}')

# Save the search space as a smi file
out_file = Path('search_space.smi')
with out_file.open('w') as fp:
    for smiles in search_qm9['smiles_0']:
        print(smiles, file=fp)
print(f'Saved search space to {out_file}')
