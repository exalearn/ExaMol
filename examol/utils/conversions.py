"""Converting between different descriptions of molecules"""
import logging
from io import StringIO

from ase import Atoms, io
from rdkit import Chem
import networkx as nx

from examol.utils.chemistry import parse_from_molecule_string

logger = logging.getLogger(__name__)


def convert_rdkit_to_nx(mol: 'Chem.Mol') -> nx.Graph:
    """Convert a networkx graph to a RDKit Molecule

    Args:
        mol (Chem.RWMol): Molecule to be converted
    Returns:
        (nx.Graph) Graph format of the molecule
    """

    graph = nx.Graph()

    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(),
                       atomic_num=atom.GetAtomicNum(),
                       formal_charge=atom.GetFormalCharge(),
                       chiral_tag=atom.GetChiralTag(),
                       hybridization=atom.GetHybridization(),
                       num_explicit_hs=atom.GetNumExplicitHs(),
                       is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       bond_type=bond.GetBondType())
    return graph


def convert_nx_to_rdkit(graph: nx.Graph) -> 'Chem.Mol':
    """Convert a networkx graph to a RDKit Molecule

    Args:
        graph (nx.Graph) Graph format of the molecule
    Returns:
        (Chem.RWMol): Molecule to be converted
    """
    mol = Chem.RWMol()

    # Special case: empty-graph
    if graph is None:
        return mol

    atomic_nums = nx.get_node_attributes(graph, 'atomic_num')
    chiral_tags = nx.get_node_attributes(graph, 'chiral_tag')
    formal_charges = nx.get_node_attributes(graph, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(graph, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(graph, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(graph, 'num_explicit_hs')
    node_to_idx = {}
    for node in graph.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(graph, 'bond_type')
    for edge in graph.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol


def convert_string_to_nx(mol_string: str) -> nx.Graph:
    """Compute a networkx graph from a SMILES string

    Args:
        mol_string: InChI or SMILES string to be parsed
    Returns:
        (nx.Graph) NetworkX representation of the molecule
    """

    # Accept either an InChI or SMILES string
    mol = parse_from_molecule_string(mol_string)
    mol = Chem.AddHs(mol)

    return convert_rdkit_to_nx(mol)


def convert_nx_to_smiles(graph: nx.Graph) -> str:
    """Compute a SMILES string from a networkx graph"""
    mol = convert_nx_to_rdkit(graph)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def write_to_string(atoms: Atoms, fmt: str, **kwargs) -> str:
    """Write an ASE atoms object to string

    Args:
        atoms: Structure to write
        fmt: Target format
        kwargs: Passed to the write function
    Returns:
        Structure written in target format
    """

    out = StringIO()
    atoms.write(out, fmt, **kwargs)
    return out.getvalue()


def read_from_string(atoms_msg: str, fmt: str) -> Atoms:
    """Read an ASE atoms object from a string

    Args:
        atoms_msg: String format of the object to read
        fmt: Format (cannot be autodetected)
    Returns:
        Parsed atoms object
    """

    out = StringIO(str(atoms_msg))  # str() ensures that Proxies are resolved
    return io.read(out, format=fmt)
