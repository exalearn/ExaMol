"""Functions needed to start evaluating a certain molecule"""
from threading import Lock
from io import StringIO
import logging

from ase.io.xyz import simple_read_xyz, write_xyz
from rdkit.Chem import AllChem
from rdkit import Chem
import networkx as nx
import numpy as np

from examol.utils.chemistry import parse_from_molecule_string
from examol.utils.conversions import convert_string_to_nx

logger = logging.getLogger(__name__)
_generate_lock = Lock()  # RDKit is not completely thread-safe


def generate_inchi_and_xyz(mol_string: str, special_cases: bool = True) -> tuple[str, str]:
    """Generate the XYZ coordinates and InChI string for a molecule using a standard procedure:

    1. Generate 3D coordinates with RDKit. Use a set random number seed
    2. Assign yet-undetermined stereochemistry based on the 3D geometry
    3. Generate an InCHi string for the molecules

    If allowed, then perform post-processing steps for common mistakes in generating geometries:

    1. Ensure cyclopropenyl groups are planar

    Args:
        mol_string: SMILES or InChI string
        special_cases: Whether to perform the post-processing
    Returns:
        - InChI string for the molecule
        - XYZ coordinates for the molecule
    """

    with _generate_lock:
        # Generate 3D coordinates for the molecule
        mol = parse_from_molecule_string(mol_string)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=1)
        AllChem.MMFFOptimizeMolecule(mol)

        # Generate an InChI string with stereochemistry information
        AllChem.AssignStereochemistryFrom3D(mol)
        inchi = Chem.MolToInchi(mol)

        # Save geometry as 3D coordinates
        xyz = f"{mol.GetNumAtoms()}\n"
        xyz += inchi + "\n"
        conf = mol.GetConformer()
        for i, a in enumerate(mol.GetAtoms()):
            s = a.GetSymbol()
            c = conf.GetAtomPosition(i)
            xyz += f"{s} {c[0]} {c[1]} {c[2]}\n"

        # Special cases for odd kinds of molecules
        if special_cases:
            xyz = fix_cyclopropenyl(xyz, mol_string)

        return inchi, xyz


def fix_cyclopropenyl(xyz: str, mol_string: str) -> str:
    """Detect cyclopropenyl groups and assure they are planar.
    Args:
        xyz: Current structure in XYZ format
        mol_string: SMILES or InChI string of the molecule
    Returns:
        Version of atoms with the rings flattened
    """

    # Find cyclopropenyl groups
    mol = parse_from_molecule_string(mol_string)
    rings = mol.GetSubstructMatches(Chem.MolFromSmarts("c1c[c+]1"))
    if len(rings) == 0:
        return xyz  # no changes

    # For each ring, flatten it
    atoms = next(simple_read_xyz(StringIO(xyz), slice(None)))
    g = convert_string_to_nx(mol_string)
    for ring in rings:
        # Get the normal of the ring
        normal = np.cross(*np.subtract(atoms.positions[ring[:2], :], atoms.positions[ring[2], :]))
        normal /= np.linalg.norm(normal)

        # Adjust the groups attached to each member of the ring
        for ring_atom in ring:
            # Get the ID of the group bonded to it
            bonded_atom = next(r for r in g[ring_atom] if r not in ring)

            # Determine the atoms that are part of that functional group
            h = g.copy()
            h.remove_edge(ring_atom, bonded_atom)
            a, b = nx.connected_components(h)
            mask = np.zeros((len(atoms),), dtype=bool)
            if bonded_atom in a:
                mask[list(a)] = True
            else:
                mask[list(b)] = True

            # Get the rotation angle
            bond_vector = atoms.positions[bonded_atom, :] - atoms.positions[ring_atom, :]
            angle = np.dot(bond_vector, normal) / np.linalg.norm(bond_vector)
            rot_angle = np.arccos(angle) - np.pi / 2
            logger.debug(f'Rotating by {rot_angle} radians')

            # Perform the rotation
            rotation_axis = np.cross(bond_vector, normal)
            atoms._masked_rotate(atoms.positions[ring_atom], rotation_axis, rot_angle, mask)

            # make the atom at a 150 angle with the ring too
            another_ring = next(r for r in ring if r != ring_atom)
            atoms.set_angle(another_ring, ring_atom, bonded_atom, 150, mask=mask)
            assert np.isclose(atoms.get_angle(another_ring, ring_atom, bonded_atom), 150).all()

            # Make sure it worked
            bond_vector = atoms.positions[bonded_atom, :] - atoms.positions[ring_atom, :]
            angle = np.dot(bond_vector, normal) / np.linalg.norm(bond_vector)
            final_angle = np.arccos(angle)
            assert np.isclose(final_angle, np.pi / 2).all()

        logger.info(f'Detected {len(rings)} cyclopropenyl rings. Ensured they are planar.')

        # Write to a string
        out = StringIO()
        write_xyz(out, [atoms])
        return out.getvalue()
