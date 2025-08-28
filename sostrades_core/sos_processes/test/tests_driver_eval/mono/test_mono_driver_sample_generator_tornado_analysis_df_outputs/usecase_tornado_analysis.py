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
import pandas as pd

from sostrades_core.sos_processes.test.test_disc_all_types.usecase import Study as Study_disc_all_types
from sostrades_core.study_manager.study_manager import StudyManager


class Study(StudyManager):
    def __init__(self, run_usecase=False, execution_engine=None) -> None:
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)

    def setup_usecase(self):
        """Usecase for lhs DoE and Eval on x variable of Sellar Problem"""
        ns = f"{self.study_name}"

        input_selection_a = {
            "selected_input": [True],
            "full_name": ["Coupling.DiscAllTypes.z"],
        }
        input_selection_a = pd.DataFrame(input_selection_a)

        output_selection_ind = {
            "selected_output": [True, True, True],
            "full_name": ["Coupling.DiscAllTypes.df_out", "Coupling.DiscAllTypes.o", "Coupling.DiscAllTypes.dict_out"],
        }
        output_selection_ind = pd.DataFrame(output_selection_ind)
        sub_study = Study_disc_all_types(self.ee)

        sub_study.study_name = f'{self.study_name}.Eval.Coupling'
        disc_dict = sub_study.setup_usecase()
        disc_dict[f'{ns}.SampleGenerator.sampling_method'] = "tornado_chart_analysis"
        disc_dict[f'{ns}.SampleGenerator.variation_list'] = [-10.0, 10.0]
        disc_dict[f'{ns}.Eval.with_sample_generator'] = True
        disc_dict[f'{ns}.SampleGenerator.eval_inputs'] = input_selection_a
        disc_dict[f'{ns}.Eval.gather_outputs'] = output_selection_ind

        # Disc1 inputs
        # disc_dict[f"{ns}.Eval.x"] = 10.0
        # disc_dict[f"{ns}.Eval.Coupling.Disc1.a"] = 1.0
        # disc_dict[f"{ns}.Eval.Coupling.Disc1.b"] = 100.0
        # disc_dict[f"{ns}.y"] = 4.0
        # disc_dict[f"{ns}.Eval.Coupling.Disc1.indicator"] = 53.0

        return [disc_dict]


if "__main__" == __name__:
    ns = "usecase_tornado_analysis"
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    uc_cls.run()


    from sostrades_core.tools.post_processing.post_processing_factory import (
        PostProcessingFactory,
    )

    post_processing_factory = PostProcessingFactory()
    for disc in uc_cls.execution_engine.root_process.proxy_disciplines:
        if disc.sos_name.endswith('tornado_chart_analysis'):
            filters = post_processing_factory.get_post_processing_filters_by_discipline(disc)
            graph_list = post_processing_factory.get_post_processing_by_discipline(disc, filters, as_json=False)

            for chart in graph_list:
                chart.to_plotly().show()

