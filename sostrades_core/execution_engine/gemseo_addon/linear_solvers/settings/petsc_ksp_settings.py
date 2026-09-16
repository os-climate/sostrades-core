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

from os import getenv
from collections.abc import Callable
from pydantic import Field, NonNegativeFloat,AliasChoices,PositiveInt,WithJsonSchema
from strenum import StrEnum
from gemseo.typing import StrKeyMapping  # noqa: TC002
from typing import Annotated

if getenv("USE_PETSC", "").lower() in ("true", "1"):
    from gemseo_petsc.linear_solvers.settings.petsc_ksp_settings import BasePetscKSPSettings

    """Settings for the SoSTrades wrapper of PETSc KSP linear solvers."""


    class SoSPreconditionerType(StrEnum):
        """
        The type of the precondtioner.

        See
        [https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html].
        """

        JACOBI = "jacobi"
        BJACOBI = "bjacobi"
        SOR = "sor"
        EISENSTAT = "eisenstat"
        ICC = "icc"
        ILU = "ilu"
        ASM = "asm"
        GASM = "gasm"
        GAMG = "gamg"
        BDDC = "bddc"
        KSP = "ksp"
        COMPOSITE = "composite"
        LU = "lu"
        CHOLESKY = "cholesky"
        NONE = "none"
        SHELL = "shell"



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



        ksp_pre_processor: Annotated[Callable, WithJsonSchema({})] | None = Field(
                default=None,
                description="""A callback function that is called before calling ksp.solve().

        The function is called with (KSP problem, options dict) as arguments.
        It allows the user to obtain an advanced configuration
        that is not supported by the current wrapper.
        If None, do not perform any call.""",
            )

        maxiter: PositiveInt = Field(
                default=100_000,
                validation_alias=AliasChoices("max_iter", "maxiter"),
                description="Maximum number of iterations.",
            )

        monitor_residuals: bool = Field(
                default=False,
                description="""Whether to store the residuals during convergence.

        WARNING: as said in Petsc documentation,
         "the routine is slow and should be used only for testing or convergence studies,
         not for timing."
        """,
            )

        options_cmd: StrKeyMapping | None = Field(
                default=None,
                description="""The options to pass to the PETSc KSP solver.

        If None, use the default options.""",
            )

        set_from_options: bool = Field(
                default=False, description="""Whether the options are set from sys.argv."""
            )

        view_config: bool = Field(
                default=False,
                description="""Whether to view the configuration of the solver before run.

        Configuration is viewed by calling ksp.view().""",
            )

