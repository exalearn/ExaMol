"""Functions needed to start evaluating a certain molecule"""
from threading import Lock
from io import StringIO
import logging

from ase.io.xyz import simple_read_xyz, write_xyz
from rdkit.Chem import AllChem, rdDetermineBonds
from rdkit import Chem
import networkx as nx
import numpy as np

from examol.store.models import MoleculeRecord, Conformer, EnergyEvaluation
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

        # Save the conformer to an XYZ file
        xyz = write_xyz_from_mol(mol, inchi)

        # Special cases for odd kinds of molecules
        if special_cases:
            xyz = fix_cyclopropenyl(xyz, mol_string)

        return inchi, xyz


def write_xyz_from_mol(mol: Chem.Mol, comment: str = ""):
    """Write an RDKit Mol object to an XYZ-format string

    Args:
        mol: Molecule to write
        comment: Comment line for the file
    Returns:
        XYZ-format version of the molecule
    """
    xyz = f"{mol.GetNumAtoms()}\n"
    xyz += comment + "\n"
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        c = conf.GetAtomPosition(i)
        xyz += f"{s} {c[0]} {c[1]} {c[2]}\n"
    return xyz


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


def add_initial_conformer(record: MoleculeRecord) -> MoleculeRecord:
    """Add an initial conformation to the record

    Generates an XYZ using RDKit if none are available then adds the MMF94 to all neutral conformers.

    Generated conformer is stored with a config name of ``mmff``, a charge of 0, and a source of ``optimize``.
    MMFF energies are stored using the configuration name ``mmff``.

    Args:
        record: Record to be processed
    Returns:
        Input record
    """

    # Generate XYZ if needed
    if not any(x.source == 'initial' for x in record.conformers):
        _, xyz = generate_inchi_and_xyz(record.identifier.smiles)
        if xyz is None:
            logger.warning(f'XYZ generation failed for "{record.key}"')
        else:
            record.conformers.append(
                Conformer.from_xyz(xyz, charge=0, config_name='mmff', source='initial')
            )

    # Compute MMFF94 for all neutral conformers
    for conf in record.conformers:
        # Skip if already done too
        if conf.charge != 0 or any(x.config_name == 'mmff' for x in conf.energies):
            continue

        # Load molecule from RDKit then detect bonds
        try:
            mol = Chem.MolFromXYZBlock(conf.xyz)
            rdDetermineBonds.DetermineConnectivity(mol)
            rdDetermineBonds.DetermineBonds(mol)
        except ValueError:
            continue  # Skip molecules that fail

        # Compute the MMFF94 energy then store it
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is not None:
            energy = AllChem.MMFFGetMoleculeForceField(mol, props).CalcEnergy()
            conf.energies.append(
                EnergyEvaluation(energy=energy, config_name='mmff', charge=0, solvent=None)
            )

    return record
