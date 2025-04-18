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

# from pdb_wizard import PBC
import random
from copy import deepcopy
import scipy

from mmc.pdb_wizard import PBC

BOLTZMANN = openmm.unit.BOLTZMANN_CONSTANT_kB * openmm.unit.AVOGADRO_CONSTANT_NA


def append_system(existing_system, system_to_append, cutoff, index_map=None):
    """Appends a system object onto the end of an existing system.

    Parameters
    ----------
    existing_system: openmm.System, optional
        The base system to extend.
    system_to_append: openmm.System
        The system to append.
    cutoff: openff.evaluator.unit.Quantity
        The nonbonded cutoff
    index_map: dict of int and int, optional
        A map to apply to the indices of atoms in the `system_to_append`.
        This is predominantly to be used when the ordering of the atoms
        in the `system_to_append` does not match the ordering in the full
        topology.
    """
    supported_force_types = [
        openmm.HarmonicBondForce,
        openmm.HarmonicAngleForce,
        openmm.PeriodicTorsionForce,
        openmm.NonbondedForce,
        openmm.RBTorsionForce,
        openmm.CustomNonbondedForce,
        openmm.CustomBondForce,
    ]

    number_of_appended_forces = 0
    index_offset = existing_system.getNumParticles()

    # Create an index map if one is not provided.
    if index_map is None:
        index_map = {i: i for i in range(system_to_append.getNumParticles())}

    # Append the particles.
    for index in range(system_to_append.getNumParticles()):
        index = index_map[index]
        existing_system.addParticle(system_to_append.getParticleMass(index))

    # Append the constraints
    for index in range(system_to_append.getNumConstraints()):
        index_a, index_b, distance = system_to_append.getConstraintParameters(index)

        index_a = index_map[index_a]
        index_b = index_map[index_b]

        existing_system.addConstraint(
            index_a + index_offset, index_b + index_offset, distance
        )

    # Validate the forces to append.
    for force_to_append in system_to_append.getForces():
        if type(force_to_append) in supported_force_types:
            continue

        raise ValueError(
            f"The system contains an unsupported type of "
            f"force: {type(force_to_append)}."
        )

    # Append the forces.
    for force_to_append in system_to_append.getForces():
        existing_force = None

        for force in existing_system.getForces():
            if type(force) not in supported_force_types:
                raise ValueError(
                    f"The existing system contains an unsupported type "
                    f"of force: {type(force)}."
                )

            if type(force_to_append) is not type(force):
                continue

            if isinstance(force_to_append, openmm.CustomNonbondedForce) or isinstance(
                force_to_append, openmm.CustomBondForce
            ):
                if force_to_append.getEnergyFunction() != force.getEnergyFunction():
                    continue

            existing_force = force
            break

        if existing_force is None:
            if isinstance(force_to_append, openmm.CustomNonbondedForce):
                existing_force = openmm.CustomNonbondedForce(
                    force_to_append.getEnergyFunction()
                )
                existing_force.setCutoffDistance(cutoff)
                existing_force.setNonbondedMethod(
                    openmm.CustomNonbondedForce.CutoffPeriodic
                )
                for index in range(force_to_append.getNumGlobalParameters()):
                    existing_force.addGlobalParameter(
                        force_to_append.getGlobalParameterName(index),
                        force_to_append.getGlobalParameterDefaultValue(index),
                    )
                for index in range(force_to_append.getNumPerParticleParameters()):
                    existing_force.addPerParticleParameter(
                        force_to_append.getPerParticleParameterName(index)
                    )
                existing_system.addForce(existing_force)

            elif isinstance(force_to_append, openmm.CustomBondForce):
                existing_force = openmm.CustomBondForce(
                    force_to_append.getEnergyFunction()
                )
                for index in range(force_to_append.getNumGlobalParameters()):
                    existing_force.addGlobalParameter(
                        force_to_append.getGlobalParameterName(index),
                        force_to_append.getGlobalParameterDefaultValue(index),
                    )
                for index in range(force_to_append.getNumPerBondParameters()):
                    existing_force.addPerBondParameter(
                        force_to_append.getPerBondParameterName(index)
                    )
                existing_system.addForce(existing_force)

            else:
                existing_force = type(force_to_append)()
                existing_system.addForce(existing_force)

        if isinstance(force_to_append, openmm.HarmonicBondForce):
            # Add the bonds.
            for index in range(force_to_append.getNumBonds()):
                index_a, index_b, *parameters = force_to_append.getBondParameters(index)

                index_a = index_map[index_a]
                index_b = index_map[index_b]

                existing_force.addBond(
                    index_a + index_offset, index_b + index_offset, *parameters
                )

        elif isinstance(force_to_append, openmm.HarmonicAngleForce):
            # Add the angles.
            for index in range(force_to_append.getNumAngles()):
                (
                    index_a,
                    index_b,
                    index_c,
                    *parameters,
                ) = force_to_append.getAngleParameters(index)

                index_a = index_map[index_a]
                index_b = index_map[index_b]
                index_c = index_map[index_c]

                existing_force.addAngle(
                    index_a + index_offset,
                    index_b + index_offset,
                    index_c + index_offset,
                    *parameters,
                )

        elif isinstance(force_to_append, openmm.PeriodicTorsionForce):
            # Add the torsions.
            for index in range(force_to_append.getNumTorsions()):
                (
                    index_a,
                    index_b,
                    index_c,
                    index_d,
                    *parameters,
                ) = force_to_append.getTorsionParameters(index)

                index_a = index_map[index_a]
                index_b = index_map[index_b]
                index_c = index_map[index_c]
                index_d = index_map[index_d]

                existing_force.addTorsion(
                    index_a + index_offset,
                    index_b + index_offset,
                    index_c + index_offset,
                    index_d + index_offset,
                    *parameters,
                )

        elif isinstance(force_to_append, openmm.NonbondedForce):
            # Add the vdW parameters
            for index in range(force_to_append.getNumParticles()):
                index = index_map[index]

                existing_force.addParticle(
                    *force_to_append.getParticleParameters(index)
                )

            # Add the 1-2, 1-3 and 1-4 exceptions.
            for index in range(force_to_append.getNumExceptions()):
                (
                    index_a,
                    index_b,
                    *parameters,
                ) = force_to_append.getExceptionParameters(index)

                index_a = index_map[index_a]
                index_b = index_map[index_b]

                existing_force.addException(
                    index_a + index_offset, index_b + index_offset, *parameters
                )

        elif isinstance(force_to_append, openmm.RBTorsionForce):
            # Support for RBTorisionForce needed for OPLSAA, etc
            for index in range(force_to_append.getNumTorsions()):
                torsion_params = force_to_append.getTorsionParameters(index)
                for i in range(4):
                    torsion_params[i] = index_map[torsion_params[i]] + index_offset

                existing_force.addTorsion(*torsion_params)

        elif isinstance(force_to_append, openmm.CustomNonbondedForce):
            for index in range(force_to_append.getNumParticles()):
                nb_params = force_to_append.getParticleParameters(index_map[index])
                existing_force.addParticle(nb_params)

            # Add the 1-2, 1-3 and 1-4 exceptions.
            for index in range(force_to_append.getNumExclusions()):
                (
                    index_a,
                    index_b,
                ) = force_to_append.getExclusionParticles(index)

                index_a = index_map[index_a]
                index_b = index_map[index_b]

                existing_force.addExclusion(
                    index_a + index_offset, index_b + index_offset
                )

        elif isinstance(force_to_append, openmm.CustomBondForce):
            for index in range(force_to_append.getNumBonds()):
                index_a, index_b, bond_params = force_to_append.getBondParameters(index)

                index_a = index_map[index_a] + index_offset
                index_b = index_map[index_b] + index_offset

                existing_force.addBond(index_a, index_b, bond_params)

        number_of_appended_forces += 1

    if number_of_appended_forces != system_to_append.getNumForces():
        raise ValueError("Not all forces were appended.")


def create_empty_system():
    system = openmm.System()
    system.addForce(openmm.HarmonicBondForce())
    system.addForce(openmm.HarmonicAngleForce())
    system.addForce(openmm.PeriodicTorsionForce())

    nonbonded_force = openmm.NonbondedForce()
    nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)

    system.addForce(nonbonded_force)

    return system

def monte_carlo(temperature: openmm.unit.Quantity, new_energy: openmm.unit.Quantity, old_energy: openmm.unit.Quantity) -> bool:
    delta_energy = new_energy - old_energy
    probability = np.exp(-delta_energy / (BOLTZMANN * temperature))
    random_number = random.uniform(0, 1)

    return probability >= random_number


def gas_formation_energy(
        gas: Molecule,
        ff: ForceField,
        integrator: openmm.Integrator,
        platform_name: str,
        gas_pos: openmm.unit.Quantity
) -> openmm.unit.Quantity:
    gas_mols = [deepcopy(gas)]
    gas_top = Topology.from_molecules(gas_mols)
    interchange = Interchange.from_smirnoff(topology=gas_top, force_field=ff)
    openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)

    openmm_sim = openmm.app.Simulation(
        gas_top.to_openmm(),
        openmm_sys,
        deepcopy(integrator),
        platform=openmm.Platform.getPlatformByName(platform_name),
    )

    openmm_sim.context.setPositions(gas_pos)
    openmm_sim.minimizeEnergy()
    context = openmm_sim.context.getState(getEnergy=True)
    assert (type(context.getPotentialEnergy())) == openmm.unit.Quantity
    return context.getPotentialEnergy()


def calculate_average_gas_distance(num_gases: int, positions_nm: openmm.unit.Quantity, mof_atom_count: int, atoms_per_gas: int, pbc: PBC) -> openmm.unit.Quantity:
    """
    Calculate the average minimum distance between each CO2 molecule's closest atom and all other atoms in the system, considering periodic boundary conditions.
    """
    if num_gases < 1:
        return 0.0 * openmm.unit.nanometer

    # context = self.get_context(num_gases)
    # state = context.getState(getPositions=True)
    # positions = state.getPositions(asNumpy=True)
    # positions_nm = positions.value_in_unit(openmm.unit.nanometer)

    # mof_atom_count = self._mof_top.n_atoms
    # atoms_per_gas = self._gas.n_atoms
    total_min = 0.0

    all_positions = positions_nm  # All atoms including MOF and gas

    for gas_idx in range(num_gases):
        start = mof_atom_count + gas_idx * atoms_per_gas
        end = start + atoms_per_gas
        current_co2_indices = np.arange(start, end)
        mask = np.ones(len(all_positions), dtype=bool)
        mask[current_co2_indices] = False
        other_positions = all_positions[mask]

        co2_atoms = all_positions[start:end]

        min_dist = np.inf
        for atom in co2_atoms:
            displacements = other_positions - atom
            min_image_displacements = np.array([pbc.min_image(disp) for disp in displacements])
            current_min = np.min(min_image_displacements)
            if current_min < min_dist:
                min_dist = current_min

        total_min += min_dist

    average = total_min / num_gases
    return average * openmm.unit.nanometer
