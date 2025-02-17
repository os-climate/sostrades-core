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

import pytest
from numpy import array, sqrt
from pandas import DataFrame

from sostrades_core.execution_engine.disciplines_wrappers.monte_carlo_driver_wrapper import MonteCarloDriverWrapper
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator

STUDY_NAME = "MC"

ONE_VAR_DISTRIB = {
    f"{STUDY_NAME}.Eval_MC.x": {
        MonteCarloDriverWrapper.DISTRIBUTION_TYPE_KEY: "OTTriangularDistribution",
        "minimum": 0,
        "maximum": 10,
        "mode": 5,
    }
}

TWO_VAR_DISTRIB = {
    f"{STUDY_NAME}.Eval_MC.x": {
        MonteCarloDriverWrapper.DISTRIBUTION_TYPE_KEY: "OTTriangularDistribution",
        "minimum": 0,
        "maximum": 10,
        "mode": 5,
    },
    f"{STUDY_NAME}.Eval_MC.y_1": {
        MonteCarloDriverWrapper.DISTRIBUTION_TYPE_KEY: "OTUniformDistribution",
        "minimum": -10,
        "maximum": 10,
    },
}


@pytest.mark.parametrize("distributions", [ONE_VAR_DISTRIB, TWO_VAR_DISTRIB])
@pytest.mark.parametrize("n_objectives", [1, 5])
@pytest.mark.parametrize("criterion", ["n_samples", "target_cv", "target_std"])
def test_monte_carlo_sellar(distributions, n_objectives, criterion):
    """Test the Monte Carlo driver on the Sellar problem."""
    target_std = 2
    target_cv = 0.025
    exec_eng = ExecutionEngine(STUDY_NAME)
    factory = exec_eng.factory

    mc_builder = factory.get_builder_from_process(
        repo="sostrades_core.sos_processes.test.tests_driver_eval.monte_carlo", mod_id="test_sellar"
    )
    exec_eng.factory.set_builders_to_coupling_builder(mc_builder)
    exec_eng.configure()

    selected_outputs = [False, False, True, False, False] if n_objectives == 1 else [True] * 5
    max_n_samples = 100 if criterion == "n_samples" else 10000
    input_dict = {
        f"{STUDY_NAME}.Eval_MC.{ProxyDriverEvaluator.GATHER_OUTPUTS}": DataFrame({
            "selected_output": selected_outputs,
            "full_name": ["c_1", "c_2", "obj", "y_1", "y_2"],
        }),
        f"{STUDY_NAME}.Eval_MC.{MonteCarloDriverWrapper.SoSInputNames.INPUT_DISTRIBUTIONS}": distributions,
        f"{STUDY_NAME}.Eval_MC.{MonteCarloDriverWrapper.SoSInputNames.N_SAMPLES}": max_n_samples,
    }
    if criterion == "target_cv":
        input_dict.update({f"{STUDY_NAME}.Eval_MC.{MonteCarloDriverWrapper.SoSInputNames.TARGET_CV}": target_cv})
    elif criterion == "target_std":
        input_dict.update({f"{STUDY_NAME}.Eval_MC.{MonteCarloDriverWrapper.SoSInputNames.TARGET_STD}": target_std})

    # Sellar inputs
    input_dict[f"{STUDY_NAME}.Eval_MC.x"] = array([1.0])
    input_dict[f"{STUDY_NAME}.Eval_MC.y_1"] = array([1.0])
    input_dict[f"{STUDY_NAME}.Eval_MC.y_2"] = array([1.0])
    input_dict[f"{STUDY_NAME}.Eval_MC.z"] = array([1.0, 1.0])
    input_dict[f"{STUDY_NAME}.Eval_MC.subprocess.Sellar_Problem.local_dv"] = 10.0

    exec_eng.load_study_from_input_dict(input_dict)

    exp_tv_list = [
        f"Nodes representation for Treeview {STUDY_NAME}",
        f"|_ {STUDY_NAME}",
        "\t|_ Eval_MC",
        "\t\t|_ subprocess",
        "\t\t\t|_ Sellar_Problem",
        "\t\t\t|_ Sellar_2",
        "\t\t\t|_ Sellar_1",
    ]
    exp_tv_str = "\n".join(exp_tv_list)
    exec_eng.display_treeview_nodes(True)
    assert exp_tv_str == exec_eng.display_treeview_nodes(exec_display=True)

    exec_eng.execute()
    mc_disc = exec_eng.dm.get_disciplines_with_name("MC.Eval_MC")[0].discipline_wrapp.discipline.sos_wrapp
    input_samples = mc_disc.get_sosdisc_outputs(MonteCarloDriverWrapper.SoSOutputNames.INPUT_SAMPLES)
    output_samples = mc_disc.get_sosdisc_outputs(MonteCarloDriverWrapper.SoSOutputNames.OUTPUT_SAMPLES)

    assert list(input_samples.keys()) == list(distributions.keys())
    assert output_samples.shape[1] == n_objectives
    if criterion == "n_samples":
        assert output_samples.shape[0] == max_n_samples
    else:
        n_samples = output_samples.shape[0]
        assert n_samples < max_n_samples
        samples_std = output_samples.std(axis=0) / sqrt(n_samples)
        if criterion == "target_std":
            assert all(samples_std <= target_std)
        elif criterion == "target_cv":
            samples_mean = output_samples.mean(axis=0)
            assert all(samples_std / samples_mean <= target_cv)


if __name__ == "__main__":
    test_monte_carlo_sellar(ONE_VAR_DISTRIB, 1, "target_std")
