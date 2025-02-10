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

from typing import Any, ClassVar

from sostrades_core.execution_engine.disciplines_wrappers.monte_carlo_driver_wrapper import MonteCarloDriverWrapper
from sostrades_core.execution_engine.proxy_driver_evaluator import ProxyDriverEvaluator
from sostrades_core.execution_engine.proxy_mono_instance_driver import ProxyMonoInstanceDriver


class ProxyMonteCarloDriver(ProxyMonoInstanceDriver):
    """The proxy for the Monte Carlo driver."""

    _ontology_data: ClassVar = {
        'label': 'Monte Carlo Driver',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    REF_DISCIPLINE_NAME: str = "Eval_MC"

    DESC_IN: ClassVar[dict[str, Any]] = {
        ProxyDriverEvaluator.GATHER_OUTPUTS: {
            ProxyDriverEvaluator.TYPE: 'dataframe',
            ProxyDriverEvaluator.DATAFRAME_DESCRIPTOR: {
                'selected_output': ('bool', None, True),
                'full_name': ('string', None, False),
                'output_name': ('multiple', None, True),
            },
            ProxyDriverEvaluator.DATAFRAME_EDITION_LOCKED: False,
            ProxyDriverEvaluator.STRUCTURING: True,
        },
        MonteCarloDriverWrapper.SoSInputNames.batch_size: {
            ProxyDriverEvaluator.TYPE: 'int',
            ProxyDriverEvaluator.DEFAULT: 0,
        },
        MonteCarloDriverWrapper.SoSInputNames.input_distributions: {
            ProxyDriverEvaluator.TYPE: 'dict',
        },
        MonteCarloDriverWrapper.SoSInputNames.n_processes: {
            ProxyDriverEvaluator.TYPE: 'int',
            ProxyDriverEvaluator.DEFAULT: 1,
        },
        MonteCarloDriverWrapper.SoSInputNames.n_samples: {
            ProxyDriverEvaluator.TYPE: 'int',
            ProxyDriverEvaluator.DEFAULT: 1000,
        },
        MonteCarloDriverWrapper.SoSInputNames.target_cv: {
            ProxyDriverEvaluator.TYPE: 'float',
            ProxyDriverEvaluator.DEFAULT: 0,
        },
        MonteCarloDriverWrapper.SoSInputNames.target_std: {
            ProxyDriverEvaluator.TYPE: 'float',
            ProxyDriverEvaluator.DEFAULT: 0,
        },
        MonteCarloDriverWrapper.SoSInputNames.wait_time_between_samples: {
            ProxyDriverEvaluator.TYPE: 'float',
            ProxyDriverEvaluator.DEFAULT: 0.0,
        },
    }

    DESC_OUT: ClassVar[dict[str, Any]] = {
        MonteCarloDriverWrapper.SoSOutputNames.input_samples: {
            ProxyDriverEvaluator.TYPE: 'dataframe',
            'unit': None,
        },
        MonteCarloDriverWrapper.SoSOutputNames.output_samples: {
            ProxyDriverEvaluator.TYPE: 'dataframe',
            'unit': None,
        },
    }

    def update_data_io_with_subprocess_io(self):
        """
        Update the DriverEvaluator _data_in with subprocess i/o.
        Overload the ProxyDriverEvaluator method to avoid adding the subprocess outputs.
        """
        self._restart_data_io_to_disc_io()

    def setup_sos_disciplines(self) -> None:
        """Overload the ProxyMonoInstanceDriver method to avoid adding irrelevant outputs."""
