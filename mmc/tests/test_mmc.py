"""
Unit and regression test for the mmc package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import mmc
from mmc.simulation import Simulation
from pydantic import BaseModel, FilePath
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.Draw import IPythonConsole
from openff.toolkit import Molecule, Topology, ForceField
from openff.interchange import Interchange
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
import openmm
from openmm import LocalEnergyMinimizer
import numpy as np
import sys
import random
from copy import deepcopy
import scipy


def test_mmc_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "mmc" in sys.modules

# No kill like over kill
METALS = [3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25 ,26, 27, 28, 29, 30, 31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

def test_simulation():
    random.seed(42)
    mol = Chem.MolFromXYZFile('mmc/data/HKUST-1.xyz')
    rdDetermineBonds.DetermineConnectivity(mol)

    editable = Chem.EditableMol(mol)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a1_idx = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtom()
        a2_idx = bond.GetEndAtomIdx()
        if a1.GetAtomicNum() in METALS or a2.GetAtomicNum() in METALS:
            editable.RemoveBond(a1_idx, a2_idx)

    frags = Chem.GetMolFrags(editable.GetMol(), asMols=True, sanitizeFrags=True)
    for frag in frags:
        if frag.GetNumAtoms() == 1:
            charge = 2
        else:
            charge = -3
    rdDetermineBonds.DetermineBonds(frag, charge=charge)

    # mols = [Molecule.from_rdkit(frag) for frag in frags]
    #
    # for mol in mols:
    #     if mol.to_smiles() == '[Cu]':
    #         mol.atom(0).formal_charge = 2

