from mmc.utils import BOLTZMANN, append_system, build_context
from mmc.pdb_wizard import PBC
from pydantic import BaseModel, FilePath
from typing import List
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


class Simulation(BaseModel):
    model_config = {
        # Will probably remove when users no longer have to pass in a list of Molecule for the mof
        'arbitrary_types_allowed': True  
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

    def __init__(self, **data):
        super().__init__(**data)
        self.gas = Molecule.from_smiles(self.gas_smiles)
        self.ff = ForceField(self.ff_path)
        self.pbc = PBC(
            self.box_dim.value_in_unit(openmm.unit.nanometer),
            self.box_dim.value_in_unit(openmm.unit.nanometer),
            self.box_dim.value_in_unit(openmm.unit.nanometer),
            90, 90, 90
        )

        self.mof_top = Topology.from_molecules(self.mof)
        self.mof_top.box_vectors = np.array([[self.box_dim, 0, 0], [0, self.box_dim, 0], [0, 0, self.box_dim]]) * unit.nanometer
        self.mof_top.is_periodic = self.is_periodic

        interchange = Interchange.from_smirnoff(
            topology=self.mof_top, force_field=self.ff
        )
        self.mof_openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)
        for i in range(self.mof_openmm_sys.getNumParticles()):
            self.mof_openmm_sys.setParticleMass(i, 0.0)

        if self.platform_name not in ["CPU", "GPU"]:
            raise Exception("Invalid platform name. Must be 'CPU' or 'GPU'")

        self.contexts = {0: build_context(0, self.mof_top.get_positions().m)}

    def gas_formation_energy(self):
        gas_mols = [deepcopy(self.gas)]
        gas_top = Topology.from_molecules(gas_mols)
        interchange = Interchange.from_smirnoff(topology=gas_top, force_field=self.ff)
        openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)

        openmm_sim = openmm.app.Simulation(
            gas_top.to_openmm(),
            openmm_sys,
            self.integrator,
            platform=openmm.Platform.getPlatform(self.platform_name),
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

    def build_context(self, num_gases, positions) -> openmm.Context:
        mof_openmm_sys = deepcopy(self.mof_openmm_sys)

        if num_gases != 0:
            openff_top = deepcopy(self.mof_top)
            gas_mols = [deepcopy(self.gas) for _ in range(num_gases)]
            openff_top.add_molecules(gas_mols)
            interchange = Interchange.from_smirnoff(
                topology=openff_top, force_field=self.ff
            )
            openmm_sys = interchange.to_openmm(combine_nonbonded_forces=False)

            context = openmm.Context(
                openmm_sys, self.integrator, openmm.Platform.getPlatformByName("CPU")
            )
            context.setPositions(positions)
        else:
            context = openmm.Context(
                mof_openmm_sys,
                self.integrator,
                openmm.Platform.getPlatformByName("CPU"),
            )
            context.setPositions(positions)

        return context

    def system_energy(self, gases: np.ndarray, minimize_energy: bool) -> float:
        mof_openmm_sys = deepcopy(self.mof_openmm_sys)
        positions = deepcopy(self.mof_top.get_positions().m)

        if len(gases) != 0:
            gas_mols = [
                deepcopy(self.gas) for _ in range(len(gases) // len(self.gas.atoms))
            ]
            gas_top = Topology.from_molecules(gas_mols)
            gas_interchange = Interchange.from_smirnoff(
                topology=gas_top, force_field=self.ff
            )
            gas_openmm_sys = gas_interchange.to_openmm(combine_nonbonded_forces=False)

            # Create modeller with MOF topology and positions
            modeller = openmm.app.Modeller(
                self.mof_top.to_openmm(), to_openmm(positions * unit.nanometer)
            )

            # Add gas topology and positions
            gas_positions = ([x.m for x in gases] * openmm.unit.nanometer)
            modeller.add(gas_top.to_openmm(), gas_positions)

            # Get merged topology and positions
            merged_top = modeller.topology
            merged_positions = modeller.positions

            # Combine force systems
            append_system(mof_openmm_sys, gas_openmm_sys, 1 * unit.nanometer)

            openmm_integrator = self.integrator
            openmm_sim = openmm.app.Simulation(
                merged_top,
                mof_openmm_sys,
                openmm_integrator,
                platform=openmm.Platform.getPlatformByName("CPU"),
            )
            openmm_sim.context.setPositions(merged_positions)
        else:
            openmm_integrator = self.integrator
            openmm_sim = openmm.app.Simulation(
                self.mof_top.to_openmm(),
                mof_openmm_sys,
                openmm_integrator,
                platform=openmm.Platform.getPlatformByName("CPU"),
            )
            openmm_sim.context.setPositions(to_openmm(positions * unit.nanometer))

        pdb_reporter = openmm.app.PDBReporter("out.pdb", 1)
        openmm_sim.reporters.append(pdb_reporter)

        if minimize_energy:
            openmm_sim.minimizeEnergy()
        # openmm_sim.step(1)

        context = openmm_sim.context.getState(getEnergy=True)
        return context.getPotentialEnergy()

    def suggest_gas_position(self, current_positions):
        while True:
            new_pos = deepcopy(self.gas_pos)
            rotation_matrix = scipy.spatial.transform.Rotation.random().as_matrix()
            shift = np.array(
                [
                    random.uniform(0, self.abc[0].value_in_unit(openmm.unit.nanometer)),
                    random.uniform(0, self.abc[1].value_in_unit(openmm.unit.nanometer)),
                    random.uniform(0, self.abc[2].value_in_unit(openmm.unit.nanometer)),
                ]
            )  # nm
            new_pos = np.dot(new_pos, rotation_matrix) + shift
            valid = True

            for atom in new_pos:
                for dim in atom:
                    if dim < 0 or dim > max(self.abc):
                        valid = False

            for existing_atom in current_positions.value_in_unit(openmm.unit.nanometer):
                for new_atom in new_pos:
                    if self.pbc.min_image(
                        (existing_atom - new_atom)
                    ) < self.r_cutoff.value_in_unit(openmm.unit.nanometer):
                        valid = False

            if not valid:
                continue

            return new_pos
    
    def get_context(self, num_gases: int) -> openmm.Context:
        if num_gases not in contexts:
            if num_gases - 1 not in contexts:
                print("This function uses the next number smaller as the basis for the next context, and for some reason the context for the previous number of gases is not available.")
                assert num_gases - 1 in contexts

            previous_context = contexts[num_gases - 1]
            previous_state = previous_context.getState(getPositions=True)
            previous_positions = previous_state.getPositions(asNumpy=True)

            new_positions = deepcopy(previous_positions)
            new_gas_position = suggest_gas_position(new_positions)
            total_positions = np.vstack((new_positions.value_in_unit(openmm.unit.nanometer), new_gas_position))

            new_context = build_context(num_gases, total_positions)
            contexts[num_gases] = new_context
        return contexts[num_gases]

    def simulate(self, steps: int) -> float:
        # gases = []  # Must be a list where elements are Quantity<[x, y, z] * unit.nanometer> coordinates
        num_gases = 0
        old_energy = get_context(0, contexts).getState(getEnergy=True).getPotentialEnergy()


        for timestep in range(1):
            old_context = get_context(num_gases, contexts)
            operation = random.random()
            operation = 0.01
            positions = old_context.getState(getPositions=True).getPositions(asNumpy=True)[:]
            gases = old_context.getState(getPositions=True).getPositions(asNumpy=True)[len(MOF_TOP.get_positions()):]
            assert len(gases) % ATOMS_PER_GAS_MOLECULE == 0


            if operation < PROB_INSERT_DELETE:  # Insert
                print('Inserting')
                new_context = get_context(num_gases + 1, contexts)
                new_gas = suggest_gas_position(positions)
                new_pos = np.vstack((positions, new_gas))
                new_context.setPositions(new_pos)

                print(new_context.getState(getEnergy=True).getPotentialEnergy())
                LocalEnergyMinimizer.minimize(new_context)

                new_energy = new_context.getState(getEnergy=True).getPotentialEnergy() - GAS_FORMATION_ENERGY
                print(new_context.getState(getEnergy=True).getPotentialEnergy())
                print('system_energy')
                print(system_energy(([x * unit.nanometer for x in np.vstack((gases, new_gas))]), True))
                print(system_energy(([x * unit.nanometer for x in np.vstack((gases, new_gas))]), False))
                print()

                if monte_carlo_test(new_energy, old_energy):
                    print('Accepted')
                    num_gases += 1
                    old_energy = new_energy
                    contexts[num_gases] = new_context

            elif operation < 2 * PROB_INSERT_DELETE:  # Delete
                if len(gases) == 0:
                    continue
                print('Deleting')
                new_context = get_context(num_gases - 1, contexts)
                gas_to_remove = random.randint(0, (len(gases) // ATOMS_PER_GAS_MOLECULE) - 1)
                gases = np.delete(gases, [gas_to_remove * ATOMS_PER_GAS_MOLECULE + i for i in range(ATOMS_PER_GAS_MOLECULE)], 0)
                positions = np.vstack((positions[:len(MOF_TOP.get_positions())], gases))
                new_context.setPositions(positions)
                new_energy = new_context.getState(getEnergy=True).getPotentialEnergy() + GAS_FORMATION_ENERGY

                if monte_carlo_test(new_energy, old_energy):
                    print('Accepted')
                    num_gases -= 1
                    old_energy = new_energy
                    contexts[num_gases] = new_context

            else:  # Translate
                if len(gases) == 0:
                    continue
                print('Translating')
                new_context = get_context(num_gases, contexts)
                old_positions = new_context.getState(getPositions=True).getPositions(asNumpy=True)[:]

                gas_to_translate = random.randint(0, (len(gases) // ATOMS_PER_GAS_MOLECULE) - 1)
                new_gas = suggest_gas_position(positions)

                gases[gas_to_translate * ATOMS_PER_GAS_MOLECULE : (gas_to_translate + 1) * ATOMS_PER_GAS_MOLECULE] = new_gas * openmm.unit.nanometer
                positions[len(MOF_TOP.get_positions()):] = gases
                new_context.setPositions(positions)
                new_energy = new_context.getState(getEnergy=True).getPotentialEnergy()

                if monte_carlo_test(new_energy, old_energy):
                    print('Accepted')
                    old_energy = new_energy
                    contexts[num_gases] = new_context
                else:
                    new_context.setPositions(old_positions)
                    contexts[num_gases] = new_context
                
            print('Step: ', timestep)
            print('Num gas molecules:', num_gases)
            print()

