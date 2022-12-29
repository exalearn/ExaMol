from examol.utils import conversions as cnv


def test_to_nx():
    graph = cnv.convert_string_to_nx('C')
    assert len(graph) == 5

    smiles = cnv.convert_nx_to_smiles(graph)
    assert smiles == 'C'
