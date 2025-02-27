from math import inf
from mmc.utils import BOLTZMANN, append_system
from mmc.pdb_wizard import PBC
from pydantic import BaseModel, FilePath
from typing import List
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDetermineBonds
from openff.toolkit import Molecule, Topology, ForceField
from openff.interchange import Interchange
import openff
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
import openmm
from openmm import LocalEnergyMinimizer, Context
import numpy as np
import sys
import time
import random
from copy import deepcopy
import scipy


class Simulation(BaseModel):
    model_config = {
        # Will probably remove when users no longer have to pass in a list of Molecule for the mof
        "arbitrary_types_allowed": True,
    }
    mof_xyz: FilePath
    mof: List[Molecule]
    mof_charges_path: FilePath
    gas_smiles: str
    gas_pos: openmm.unit.Quantity
    ff_path: FilePath
    temperature: openmm.unit.Quantity = 298.15 * openmm.unit.kelvin
    pressure: openmm.unit.Quantity = 1.0 * openmm.unit.atmospheres
    timestep: openmm.unit.Quantity
    prob_insert_delete: float
    r_cutoff: openmm.unit.Quantity = 0.1 * openmm.unit.nanometer
    box_dim: openmm.unit.Quantity
    is_periodic: bool = True
    integrator: openmm.Integrator
    platform_name: str

    _gas: Molecule
    _ff: ForceField
    _pbc: PBC
    _mof_top: Topology
    _mof_openmm_sys: openmm.System
    _contexts: dict[int, Context]

    def __init__(self, **data):
        super().__init__(**data)
        self._gas = Molecule.from_smiles(self.gas_smiles)
        self._ff = ForceField(self.ff_path)
        self._pbc = PBC(
            self.box_dim.value_in_unit(openmm.unit.nanometer),
            self.box_dim.value_in_unit(openmm.unit.nanometer),
            self.box_dim.value_in_unit(openmm.unit.nanometer),
            90,
            90,
            90,
        )

        self._mof_top = Topology.from_molecules(self.mof)
        self._mof_top.box_vectors = (
            np.array([
                [self.box_dim.value_in_unit(openmm.unit.nanometer), 0, 0],
                [0, self.box_dim.value_in_unit(openmm.unit.nanometer), 0],
                [0, 0, self.box_dim.value_in_unit(openmm.unit.nanometer)]
            ]) * unit.nanometer
        )
        self._mof_top.is_periodic = self.is_periodic

        interchange = Interchange.from_smirnoff(
            topology=self._mof_top, force_field=self._ff
        )
        self._mof_openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)
        for i in range(self._mof_openmm_sys.getNumParticles()):
            self._mof_openmm_sys.setParticleMass(i, 0.0)

        if self.platform_name not in ["CPU", "GPU"]:
            raise Exception("Invalid platform name. Must be 'CPU' or 'GPU'")

        self._contexts = {0: self.build_context(0, self._mof_top.get_positions().m)}


    def gas_formation_energy(self):
        gas_mols = [deepcopy(self._gas)]
        gas_top = Topology.from_molecules(gas_mols)
        interchange = Interchange.from_smirnoff(topology=gas_top, force_field=self._ff)
        openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)

        openmm_sim = openmm.app.Simulation(
            gas_top.to_openmm(),
            openmm_sys,
            deepcopy(self.integrator),
            platform=openmm.Platform.getPlatformByName(self.platform_name),
        )

        openmm_sim.context.setPositions(self.gas_pos)
        openmm_sim.minimizeEnergy()
        context = openmm_sim.context.getState(getEnergy=True)
        return context.getPotentialEnergy()

    def monte_carlo(self, new_energy, old_energy):
        delta_energy = new_energy - old_energy
        probability = np.exp(-delta_energy / (BOLTZMANN * self.temperature))
        random_number = random.uniform(0, 1)

        return probability >= random_number


    def system_energy(
        self, positions: np.ndarray, minimize_energy: bool, num_gases: int
    ) -> float:
        mof_openmm_sys = deepcopy(self._mof_openmm_sys)

        if num_gases != 0:
            gas_mols = [deepcopy(self._gas) for _ in range(num_gases)]
            system_top = deepcopy(self._mof_top)
            system_top.add_molecules(gas_mols)
            system_top.set_positions(positions * unit.nanometer)

            interchange = Interchange.from_smirnoff(
                topology=system_top, force_field=self._ff
            )
            openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)

            openmm_sim = openmm.app.Simulation(
                system_top.to_openmm(),
                openmm_sys,
                deepcopy(self.integrator),
                platform=openmm.Platform.getPlatformByName(self.platform_name),
            )
            openmm_sim.context.setPositions(positions)
        else:
            openmm_sim = openmm.app.Simulation(
                self._mof_top.to_openmm(),
                mof_openmm_sys,
                deepcopy(self.integrator),
                platform=openmm.Platform.getPlatformByName(self.platform_name),
            )
            openmm_sim.context.setPositions(positions)

        pdb_reporter = openmm.app.PDBReporter("out.pdb", 1)
        openmm_sim.reporters.append(pdb_reporter)

        if minimize_energy:
            openmm_sim.minimizeEnergy()

        context = openmm_sim.context.getState(getEnergy=True)
        return context.getPotentialEnergy()


    def suggest_gas_position(self, current_positions):
        while True:
            new_pos = deepcopy(self.gas_pos)
            rotation_matrix = scipy.spatial.transform.Rotation.random().as_matrix()
            shift = np.array(
                [
                    random.uniform(
                        0, self.box_dim.value_in_unit(openmm.unit.nanometer)
                    ),
                    random.uniform(
                        0, self.box_dim.value_in_unit(openmm.unit.nanometer)
                    ),
                    random.uniform(
                        0, self.box_dim.value_in_unit(openmm.unit.nanometer)
                    ),
                ]
            )  # nm
            new_pos = np.dot(new_pos, rotation_matrix) + shift
            valid = True

            for atom in new_pos:
                for dim in atom:
                    if dim < 0 or dim > self.box_dim.value_in_unit(
                        openmm.unit.nanometer
                    ):
                        valid = False

            for existing_atom in current_positions.value_in_unit(openmm.unit.nanometer):
                for new_atom in new_pos:
                    if self._pbc.min_image(
                        (existing_atom - new_atom)
                    ) < self.r_cutoff.value_in_unit(openmm.unit.nanometer):
                        valid = False

            if not valid:
                continue

            return new_pos


    def build_context(self, num_gases, positions) -> openmm.Context:
        mof_openmm_sys = deepcopy(self._mof_openmm_sys)

        if num_gases != 0:
            openff_top = deepcopy(self._mof_top)
            gas_mols = [deepcopy(self._gas) for _ in range(num_gases)]
            openff_top.add_molecules(gas_mols)
            interchange = Interchange.from_smirnoff(
                topology=openff_top, force_field=self._ff
            )
            openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)

            context = openmm.Context(
                openmm_sys,
                deepcopy(self.integrator),
                openmm.Platform.getPlatformByName(self.platform_name),
            )
            context.setPositions(positions)
        else:
            context = openmm.Context(
                mof_openmm_sys,
                self.integrator,
                openmm.Platform.getPlatformByName(self.platform_name),
            )
            context.setPositions(positions)

        return context


    def get_context(self, num_gases: int) -> openmm.Context:
        if num_gases not in self._contexts:
            if num_gases - 1 not in self._contexts:
                raise ValueError(
                    "This function uses the next number smaller as the basis for the next context, and for some reason the context for the previous number of gases is not available."
                )

            previous_context = self._contexts[num_gases - 1]
            previous_state = previous_context.getState(getPositions=True)
            previous_positions = previous_state.getPositions(asNumpy=True)

            new_positions = deepcopy(previous_positions)
            new_gas_position = self.suggest_gas_position(new_positions)
            total_positions = np.vstack(
                (new_positions.value_in_unit(openmm.unit.nanometer), new_gas_position)
            )

            new_context = self.build_context(num_gases, total_positions)
            self._contexts[num_gases] = new_context
        return self._contexts[num_gases]


    def simulate(self, steps: int, apply_energy_minimizer: bool) -> float:
        ATOMS_PER_GAS_MOLECULE = len(self._gas.atoms)
        GAS_FORMATION_ENERGY = self.gas_formation_energy()
        num_gases = 0
        old_energy = self.get_context(0).getState(getEnergy=True).getPotentialEnergy()

        start_time = time.time()
        step_times = []

        for timestep in range(steps):
            step_start_time = time.time()

            old_context = self.get_context(num_gases)
            operation = random.random()
            # operation = 0.01
            positions = old_context.getState(getPositions=True).getPositions(
                asNumpy=True
            )[:]
            gases = old_context.getState(getPositions=True).getPositions(asNumpy=True)[
                len(self._mof_top.get_positions()) :
            ]
            assert len(gases) % ATOMS_PER_GAS_MOLECULE == 0

            if operation < self.prob_insert_delete:  # Insert
                print("Inserting")
                new_context = self.get_context(num_gases + 1)
                new_gas = self.suggest_gas_position(positions)
                new_pos = np.vstack((positions, new_gas))
                new_context.setPositions(new_pos)

                if apply_energy_minimizer:
                    LocalEnergyMinimizer.minimize(new_context)

                new_energy = (
                    new_context.getState(getEnergy=True).getPotentialEnergy()
                    - GAS_FORMATION_ENERGY
                )

                if self.monte_carlo(new_energy, old_energy):
                    print("Accepted")
                    num_gases += 1
                    old_energy = new_energy
                    self._contexts[num_gases] = new_context

            elif operation < 2 * self.prob_insert_delete:  # Delete
                if len(gases) == 0:
                    continue
                print("Deleting")
                new_context = self.get_context(num_gases - 1)
                gas_to_remove = random.randint(
                    0, (len(gases) // ATOMS_PER_GAS_MOLECULE) - 1
                )
                gases = np.delete(
                    gases,
                    [
                        gas_to_remove * ATOMS_PER_GAS_MOLECULE + i
                        for i in range(ATOMS_PER_GAS_MOLECULE)
                    ],
                    0,
                )
                positions = np.vstack(
                    (positions[: len(self._mof_top.get_positions())], gases)
                )
                new_context.setPositions(positions)
                if apply_energy_minimizer:
                    LocalEnergyMinimizer.minimize(new_context)
                new_energy = (
                    new_context.getState(getEnergy=True).getPotentialEnergy()
                    + GAS_FORMATION_ENERGY
                )

                if self.monte_carlo(new_energy, old_energy):
                    print("Accepted")
                    num_gases -= 1
                    old_energy = new_energy
                    self._contexts[num_gases] = new_context

            else:  # Translate
                if len(gases) == 0:
                    continue
                print("Translating")
                new_context = self.get_context(num_gases)
                old_positions = new_context.getState(getPositions=True).getPositions(
                    asNumpy=True
                )[:]

                gas_to_translate = random.randint(
                    0, (len(gases) // ATOMS_PER_GAS_MOLECULE) - 1
                )
                new_gas = self.suggest_gas_position(positions)

                gases[
                    gas_to_translate
                    * ATOMS_PER_GAS_MOLECULE : (gas_to_translate + 1)
                    * ATOMS_PER_GAS_MOLECULE
                ] = (new_gas * openmm.unit.nanometer)
                positions[len(self._mof_top.get_positions()) :] = gases
                new_context.setPositions(positions)
                if apply_energy_minimizer:
                    LocalEnergyMinimizer.minimize(new_context)
                new_energy = new_context.getState(getEnergy=True).getPotentialEnergy()

                if self.monte_carlo(new_energy, old_energy):
                    print("Accepted")
                    old_energy = new_energy
                    self._contexts[num_gases] = new_context
                else:
                    new_context.setPositions(old_positions)
                    self._contexts[num_gases] = new_context

            print("Step: ", timestep)
            print("Num gas molecules:", num_gases)
            avg_distance = self.calculate_average_gas_distance(num_gases)
            print(f"Average gas distance: {avg_distance}")
            if timestep > 0:
                time_per_step = (time.time() - start_time) / timestep
                steps_per_second = 1 / time_per_step
                estimated_time_remaining = (steps - timestep - 1) * time_per_step
                print(f"Time per step: {time_per_step}")
                print(f"Steps per second: {steps_per_second}")
                print(f"Estimated time remaining (secs): {estimated_time_remaining}")
            print()

            self.write_xyz("out.xyz", num_gases)

        return num_gases


    def write_xyz(self, filename: str, num_gases: int):
        context = self.get_context(num_gases)

        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)

        positions = positions.value_in_unit(openmm.unit.angstrom)

        elements = []
        for atom in self._mof_top.atoms:
            elements.append(atom.symbol)
        # Add elements from each gas molecule
        for _ in range(num_gases):
            for atom in self._gas.atoms:
                elements.append(atom.symbol)
        # Write to the XYZ file
        with open(filename, 'w') as f:
            f.write(f"{len(elements)}\n")
            f.write("Generated by OpenMM Simulation\n")
            for element, (x, y, z) in zip(elements, positions):
                f.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")


    def calculate_average_gas_distance(self, num_gases: int) -> openmm.unit.Quantity:
        """
        Calculate the average minimum distance between each CO2 molecule's closest atom and all other atoms in the system, considering periodic boundary conditions.

        Args:
            num_gases (int): Number of CO2 molecules in the current system.

        Returns:
            float: Average minimum distance in nanometers. Returns 0.0 if there are no CO2 molecules.
        """
        if num_gases < 1:
            return 0.0

        context = self.get_context(num_gases)
        state = context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        positions_nm = positions.value_in_unit(openmm.unit.nanometer)

        mof_atom_count = self._mof_top.n_atoms
        atoms_per_gas = self._gas.n_atoms
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
                min_image_displacements = np.array([self._pbc.min_image(disp) for disp in displacements])
                current_min = np.min(min_image_displacements)
                if current_min < min_dist:
                    min_dist = current_min

            total_min += min_dist

        average = total_min / num_gases
        return average * openmm.unit.nanometer
