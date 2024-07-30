# Copyright (c) 2024 Capgemini Engineering
# All rights reserved.
#
# Created on 30/07/2024, 11:23
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The UQ study on the Sellar MDA."""

from __future__ import annotations

from sostrades_core.sos_processes.test.test_sellar_uq.sellar_doe.usecase import Study as DOEStudy


class Study(DOEStudy):
    """The main study for the UQ on the Sellar MDA."""

    UQ_NAME = "UncertaintyQuantification"

    def setup_usecase(self):
        """Setup the usecase."""
        params = super().setup_usecase()[0]
        params.update({
            f"{self.study_name}.{self.UQ_NAME}.{option}": params[
                f"{self.study_name}.{self.SAMPLE_GENERATOR_NAME}.{option}"
            ]
            for option in ["eval_inputs", "design_space"]
        })
        params.update({
            f"{self.study_name}.{self.UQ_NAME}.gather_outputs": params[f"{self.study_name}.Eval.gather_outputs"],
        })

        return [params]


if __name__ == "__main__":
    usecase = Study(run_usecase=True, log_level="DEBUG")
    usecase.load_data()
    usecase.execution_engine.display_treeview_nodes(display_variables=True)

    usecase.run(logger_level="INFO")
