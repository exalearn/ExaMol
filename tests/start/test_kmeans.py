"""
Test k-means starting algorithms
"""
from examol.start.kmeans import KMeansStarter
import pytest


def test_basic_func():
    starter = KMeansStarter(1, 2)
    smiles_list = ["C(=O)O", "CC(=O)O", "CCO", "C#N", "CC#N"]
    selected = starter.select(smiles_list, 2)
    assert len(selected) == 2
    assert all(smiles in smiles_list for smiles in selected)


def test_count_greater_than_molecules():
    starter = KMeansStarter(1, 2)
    smiles_list = ["C(=O)O", "CC(=O)O", "CCO"]
    with pytest.raises(ValueError):
        starter.select(smiles_list, 4)


def test_count_equals_one():
    starter = KMeansStarter(1, 1)
    smiles_list = ["C(=O)O", "CC(=O)O", "CCO"]
    selected = starter.select(smiles_list, 1)
    assert len(selected) == 1
    assert selected[0] in smiles_list
