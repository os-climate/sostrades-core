'''
Copyright 2022 Airbus SAS
Modifications on 2023/04/13-2024/06/28 Copyright 2023 Capgemini

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

from pathlib import Path

import pandas as pd

from sostrades_core.study_manager.study_manager import StudyManager
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory


class Study(StudyManager):
    """Study for the UQ analysis."""

    def __init__(self, run_usecase=True, execution_engine=None):  # noqa: D107
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):  # noqa: D102
        self.uncertainty_quantification = "UncertaintyQuantification"

        self.data_dir = Path(__file__).parent / "data"
        self.data_df = pd.read_csv(self.data_dir / "data_df.csv")

        self.samples_dataframe = pd.read_csv(self.data_dir / "samples_df.csv")

        input_selection = {
            "selected_input": [True, True, True],
            "full_name": ["COC", "RC", "NRC"],
            "shortest_name": ["COC", "RC", "NRC"],
        }

        output_selection = {
            "selected_output": [True, True, True],
            "full_name": ["output1", "output2", "output3"],
            "shortest_name": ["output1", "output2", "output3"],
        }
        self.input_selection = pd.DataFrame(input_selection)
        self.output_selection = pd.DataFrame(output_selection)

        dspace = pd.DataFrame({
            "variable": ["COC", "RC", "NRC"],
            "shortest_name": ["COC", "RC", "NRC"],
            "lower_bnd": [85.0, 80.0, 80.0],
            "upper_bnd": [105.0, 120.0, 120.0],
            "nb_points": [10, 10, 10],
            "full_name": ["COC", "RC", "NRC"],
        })

        data_details_df = pd.DataFrame({
            "type": ["input"] * 3 + ["output"] * 3,
            "variable": ["COC", "RC", "NRC", "output1", "output2", "output3"],
            "name": ["COC", "RC", "NRC", "output1", "output2", "output3"],
            "unit": "$",
        })
        return {
            f"{self.study_name}.{self.uncertainty_quantification}.eval_inputs": self.input_selection,
            f"{self.study_name}.{self.uncertainty_quantification}.gather_outputs": self.output_selection,
            f"{self.study_name}.{self.uncertainty_quantification}.samples_inputs_df": self.samples_dataframe,
            f"{self.study_name}.{self.uncertainty_quantification}.samples_outputs_df": self.data_df,
            f"{self.study_name}.{self.uncertainty_quantification}.design_space": dspace,
            f"{self.study_name}.{self.uncertainty_quantification}.data_details_df": data_details_df,
        }


if __name__ == "__main__":
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()

    # # display all post_proc
    post_processing_factory = PostProcessingFactory()
    all_post_processings = post_processing_factory.get_all_post_processings(
        uc_cls.execution_engine, False, as_json=False, for_test=False
    )
    for post_proc_list in all_post_processings.values():
        for chart in post_proc_list:
            for fig in chart.post_processings:
                fig.to_plotly()  # .show()
