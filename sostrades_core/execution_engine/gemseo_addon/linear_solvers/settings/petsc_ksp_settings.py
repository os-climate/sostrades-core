# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Settings for the SoSTrades wrapper of PETSc KSP linear solvers."""

from __future__ import annotations

from gemseo_petsc.linear_solvers.settings.petsc_ksp_settings import BasePetscKSPSettings
from pydantic import Field, NonNegativeFloat
from strenum import StrEnum


class SoSPreconditionerType(StrEnum):
    """The type of the precondtioner.

    See
    [https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html].
    """

    JACOBI = "jacobi"
    ILU = "ilu"
    GASM = "gasm"


class BaseSoSPetscKSPSettings(BasePetscKSPSettings):
    """The settings of the PETSc KSP algorithms.

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
