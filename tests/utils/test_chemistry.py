"""Tests for the "chemistry" utilities"""

from rdkit import Chem

from examol.utils import chemistry as chem


def test_parse():
    mol = chem.parse_from_molecule_string('O')
    inchi = Chem.MolToInchi(mol)
    assert inchi == 'InChI=1S/H2O/h1H2'

    mol = chem.parse_from_molecule_string(inchi)
    smiles = Chem.MolToSmiles(mol)
    assert smiles == 'O'


def test_charge():
    assert chem.get_baseline_charge('O') == 0
    assert chem.get_baseline_charge('[NH4+]') == 1
    assert chem.get_baseline_charge('Fc1c(F)c1=[F+]') == 1
