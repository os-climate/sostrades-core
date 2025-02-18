'''
Copyright 2025 Capgemini

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

import pytest
from numpy import array
from numpy.testing import assert_allclose
from pandas import DataFrame, read_csv

from sostrades_core.sos_processes.test.test_uncertainty_analysis.usecase import Study
from sostrades_core.sos_wrapping.analysis_discs.uncertainty_analysis import UncertaintyAnalysis

POST_NAME = "MC post"
EVAL_MC_NAME = "Eval_MC"


@pytest.mark.parametrize("n_objectives", [1, 5])
def test_uncertainty_analysis(n_objectives: int):
    """Test the UncertaintyAnalysis discipline with fixed input-output samples."""
    input_samples = read_csv(Path(__file__).parent / "data/test_96_input_samples.csv")
    output_samples = read_csv(Path(__file__).parent / "data/test_96_output_samples.csv")
    expected_stats = read_csv(Path(__file__).parent / "data/test_96_statistics.csv")
    if n_objectives == 1:
        output_samples = DataFrame(output_samples["usecase.Eval_MC.obj"])
        expected_stats = expected_stats.loc[:, ["usecase.Eval_MC.x", "usecase.Eval_MC.obj"]]
        threshold = array([5, 45])
    else:
        threshold = array([5, 0, -19.5, 45, 5, 4])
    study = Study(run_usecase=True)
    input_dict = {
        f"{study.study_name}.{EVAL_MC_NAME}.{UncertaintyAnalysis.SoSInputNames.INPUT_SAMPLES}": input_samples,
        f"{study.study_name}.{EVAL_MC_NAME}.{UncertaintyAnalysis.SoSInputNames.OUTPUT_SAMPLES}": output_samples,
        f"{study.study_name}.{POST_NAME}.{UncertaintyAnalysis.SoSInputNames.PROBABILITY_THRESHOLD}": threshold,
    }
    study.load_data(from_input_dict=input_dict)
    study.run()
    # Check the output
    dm = study.execution_engine.dm
    stats = dm.get_value(f"{study.study_name}.{POST_NAME}.{UncertaintyAnalysis.SoSOutputNames.STATISTICS}")
    assert_allclose(stats.to_numpy(), expected_stats.to_numpy())


if __name__ == "__main__":
    test_uncertainty_analysis(1)
