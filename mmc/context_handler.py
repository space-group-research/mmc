from pydantic import BaseModel
import openmm
from copy import deepcopy
from openmm import Context
import numpy as np
from openff.toolkit import ForceField, Topology, Molecule
from openff.interchange import Interchange


class ContextHandler(BaseModel):
    _contexts: dict[int, openmm.Context]
    mof_openmm_sys: openmm.System
    mof_top: Topology
    gas: Molecule
    gas_pos: openmm.unit.Quantity
    platform_name: str
    ff: ForceField
    integrator: openmm.Integrator


    def __init__(self, **data):
        super().__init__(**data)
        self._contexts = {0: self.build_context(0, self.mof_top.get_positions().m)}


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
            # This can just be anything since we'll be writing over the positions anyway after we get the context back
            new_gas_position = deepcopy(self.gas_pos)
            total_positions = np.vstack(
                (new_positions.value_in_unit(openmm.unit.nanometer), new_gas_position)
            )

            new_context = self.build_context(num_gases, total_positions)
            self._contexts[num_gases] = new_context
        return self._contexts[num_gases]
