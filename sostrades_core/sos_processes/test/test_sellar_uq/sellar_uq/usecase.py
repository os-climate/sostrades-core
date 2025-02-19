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

from sostrades_core.sos_processes.test.test_sellar_uq.sellar_doe.usecase import Study as DOEStudy


class Study(DOEStudy):
    """The main study for the UQ on the Sellar MDA."""

    UQ_NAME = "UncertaintyQuantification"

    def __init__(self, **kwargs) -> None:  # noqa: D107
        super(DOEStudy, self).__init__(__file__, **kwargs)

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
    usecase = Study(log_level="DEBUG")
    usecase.run_usecase = True
    usecase.load_data()
    usecase.execution_engine.display_treeview_nodes(display_variables=True)

    usecase.run(logger_level="INFO")
