# Import package, test suite, and other packages as needed
import sys

import pytest

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
from mmc.pdb_wizard import PBC
from mmc.simulation import Simulation
import random
from copy import deepcopy
import scipy

from mmc.utils import BOLTZMANN, append_system, monte_carlo, gas_formation_energy

def test_monte_carlo():
    # 0 delta should always accept
    assert monte_carlo(298 * openmm.unit.kelvin, 1 * openmm.unit.kilojoules_per_mole, 1 * openmm.unit.kilojoules_per_mole)
    
    # If either of these error out I'd go try the lottery. Your luck is pretty crazy.
    assert not monte_carlo(298 * openmm.unit.kelvin, 1000 * openmm.unit.kilojoules_per_mole, 0 * openmm.unit.kilojoules_per_mole)

    assert monte_carlo(298 * openmm.unit.kelvin, 0 * openmm.unit.kilojoules_per_mole, 1000 * openmm.unit.kilojoules_per_mole)


def test_gas_formation_energy():
    # Create a simple molecule (methane)
    gas = Molecule.from_smiles("O=C=O")

    # Load the OpenFF 2.0.0 force field
    ff = ForceField("openff-2.0.0.offxml")

    # Create an integrator (parameters are arbitrary for this test)
    integrator = openmm.LangevinIntegrator(298, 1, 0.002)

    gas_pos = np.array([
        (np.array([0, 0, 0])),
        (np.array([0.1163, 0, 0])),
        (np.array([0.2326, 0, 0]))
    ]) * openmm.unit.nanometer

    # Compute the gas formation energy
    energy = gas_formation_energy(gas, ff, integrator, "CPU", gas_pos)

    # Verify the returned energy is an OpenMM Quantity with correct units
    assert isinstance(energy, openmm.unit.Quantity)
    assert energy.unit == openmm.unit.kilojoule_per_mole

    # Check that the energy is a finite number (specific value depends on FF/geometry)
    print(energy)  # This value feels... low
    assert energy.value_in_unit(openmm.unit.kilojoule_per_mole) is not None
    assert np.isfinite(energy.value_in_unit(openmm.unit.kilojoule_per_mole))
