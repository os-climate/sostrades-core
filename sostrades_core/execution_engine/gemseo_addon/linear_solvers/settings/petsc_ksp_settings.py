'''
Copyright 2024 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import annotations

from gemseo_petsc.linear_solvers.settings.petsc_ksp_settings import BasePetscKSPSettings
from pydantic import Field, NonNegativeFloat
from strenum import StrEnum

"""Settings for the SoSTrades wrapper of PETSc KSP linear solvers."""


class SoSPreconditionerType(StrEnum):
    """
    The type of the precondtioner.

    See
    [https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html].
    """

    JACOBI = "jacobi"
    ILU = "ilu"
    GASM = "gasm"


class BaseSoSPetscKSPSettings(BasePetscKSPSettings):
    """
    The settings of the PETSc KSP algorithms.

    The default numerical parameters differ from gemseo_petsc.
    `_TARGET_CLASS_NAME` will be overloaded for each algorithm.
    """

    atol: NonNegativeFloat = Field(
        default=1e-200,
        description="""The absolute convergence tolerance.

Absolute tolerance of the (possibly preconditioned) residual norm.
Algorithm stops if norm(b - A @ x) <= max(rtol*norm(b), atol).""",
    )

    dtol: NonNegativeFloat = Field(
        default=1e50,
        description="""The divergence tolerance.

The amount the (possibly preconditioned) residual norm can increase.""",
    )

    preconditioner_type: SoSPreconditionerType | None = Field(
        default=SoSPreconditionerType.ILU,
        description="""The type of the precondtioner.

See [https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html]""",
    )

    rtol: NonNegativeFloat = Field(
        default=1e-200,
        description="""The relative convergence tolerance.

Relative decrease in the (possibly preconditioned) residual norm.""",
    )
