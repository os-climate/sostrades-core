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


import numpy as np
import pandas as pd

from sostrades_core.execution_engine.gather_discipline import GatherDiscipline
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    # Import pyDecision methods for multi-criteria analysis
    from pyDecision.algorithm.saw import saw_method
    from pyDecision.algorithm.topsis import topsis_method
    from pyDecision.algorithm.vikor import vikor_method
    try:
        from pyDecision.algorithm.promethee_gaia import gaia_method
    except ImportError:
        gaia_method = None
    try:
        from pyDecision.compare.compare import compare_ranks_crisp, corr_viz, plot_rank_freq
    except ImportError:
        compare_ranks_crisp = None
        corr_viz = None
        plot_rank_freq = None
    PYDECISION_AVAILABLE = True
except ImportError:
    PYDECISION_AVAILABLE = False
    topsis_method = None
    vikor_method = None
    saw_method = None
    gaia_method = None


class ValueAnalysisChart(TwoAxesInstanciatedChart):
    """Custom chart class for value analysis visualizations using Plotly"""

    def __init__(self, fig, chart_name="Value Analysis Chart"):
        # Initialize with dummy data since we'll override to_plotly
        super().__init__('', '', chart_name=chart_name)
        self.plotly_fig = fig

    def to_plotly(self, logger=None):
        """Override to return our custom Plotly figure"""
        return self.plotly_fig


class ValueAnalysisDiscipline(SoSWrapp):
    """
    Value Analysis discipline using pyDecision for multi-criteria decision analysis.
    Analyzes DoE results with multiple criteria inputs and outputs, generating
    various visualizations including GAIA charts using Plotly.
    """

    # SoSTrades post-processing module
    _postprocessing_module = "sostrades_core.sos_wrapping.value_analysis.ValueAnalysis_post_processing"

    # ontology information
    _ontology_data = {
        'label': 'Value Analysis using pyDecision',
        'type': 'Research',
        'source': 'SoSTrades Value Analysis Project',
        'version': '1.0',
    }
    GATHER_OUTPUTS_DESC = GatherDiscipline.EVAL_OUTPUTS_DESC.copy()
    GATHER_OUTPUTS_DESC['namespace']='ns_eval'
    DESC_IN = {
        GatherDiscipline.GATHER_OUTPUTS: GATHER_OUTPUTS_DESC,
        # 'top_speed_dict': {'type': 'dict', 'default': {},
        #                   'namespace': 'ns_public',
        #                   'description': 'DoE evaluation results for top_speed from SimpleDoE'},
        # 'range_dict': {'type': 'dict', 'default': {},
        #               'namespace': 'ns_public',
        #               'description': 'DoE evaluation results for range from SimpleDoE'},
        # 'manufacturing_cost_dict': {'type': 'dict', 'default': {},
        #                            'namespace': 'ns_public',
        #                            'description': 'DoE evaluation results for manufacturing_cost from SimpleDoE'},
        'input_criteria_names': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [],
                                'description': 'Names of input criteria/variables from DoE'},
        'output_criteria_names': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [],
                                 'description': 'Names of output criteria/objectives from DoE'},
        'criteria_types': {'type': 'list', 'subtype_descriptor': {'list': 'string'}, 'default': [],
                          'description': 'Type of each criterion: "max" for benefit criteria, "min" for cost criteria'},
        'criteria_weights': {'type': 'list', 'subtype_descriptor': {'list': 'float'}, 'default': [],
                            'description': 'Weights for each criterion (optional, equal weights if not provided)'},
        'mcda_methods': {'type': 'list', 'subtype_descriptor': {'list': 'string'},
                        'default': ['topsis', 'saw', 'vikor'],
                        'description': 'Multi-criteria methods to apply: topsis, saw, vikor, promethee_gaia'},
        'generate_gaia_chart': {'type': 'bool', 'default': True,
                               'description': 'Generate GAIA visualization chart'},
        'generate_correlation_chart': {'type': 'bool', 'default': True,
                                      'description': 'Generate correlation matrix chart'},
        'generate_ranking_comparison': {'type': 'bool', 'default': True,
                                       'description': 'Generate ranking comparison chart'},
        'samples_outputs_df': {'type': 'dataframe','namespace':'ns_eval'},
        'samples_inputs_df': {'type': 'dataframe', 'namespace': 'ns_eval'}
    }

    DESC_OUT = {
        'mcda_rankings': {'type': 'dataframe',
                         'description': 'Rankings from different MCDA methods'},
        'ranking_scores': {'type': 'dataframe',
                          'description': 'Scores from different MCDA methods'},
        'correlation_matrix': {'type': 'dataframe',
                              'description': 'Correlation matrix between different ranking methods'},
        'best_alternatives': {'type': 'dataframe',
                             'description': 'Top alternatives identified by each method'},
        'analysis_summary': {'type': 'dict',
                            'description': 'Summary of the multi-criteria analysis'}
    }

    def setup_sos_disciplines(self):
        """Setup the discipline"""
        if not PYDECISION_AVAILABLE:
            self.logger.warning("pyDecision is not available. Please install it using: pip install pyDecision")
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly is not available. Please install it using: pip install plotly")

        dynamic_inputs = {}
        if GatherDiscipline.GATHER_OUTPUTS in self.get_data_in():
            gather_outputs = self.get_sosdisc_inputs(GatherDiscipline.GATHER_OUTPUTS)
            if gather_outputs is not None and not gather_outputs[gather_outputs['selected_output']].empty :

                dynamic_inputs.update({f'{full_name}_dict': {'type': 'dict',
                 'namespace': 'ns_eval'} for full_name in gather_outputs[gather_outputs['selected_output']]['full_name']})
        self.add_inputs(dynamic_inputs)

    def run(self):
        """Execute the value analysis"""
        print("\n" + "="*80)
        print("[INFO] STARTING VALUE ANALYSIS EXECUTION")
        print("="*80)

        # Get inputs - automatically coupled from SimpleDoE through SoSTrades namespace mechanism
        gather_outputs = self.get_sosdisc_inputs(GatherDiscipline.GATHER_OUTPUTS)
        dict_list = gather_outputs[gather_outputs['selected_output']]['full_name']

        input_criteria_names = self.get_sosdisc_inputs('input_criteria_names')
        output_criteria_names = self.get_sosdisc_inputs('output_criteria_names')
        criteria_types = self.get_sosdisc_inputs('criteria_types')
        criteria_weights = self.get_sosdisc_inputs('criteria_weights')
        mcda_methods = self.get_sosdisc_inputs('mcda_methods')


        # Check if DoE outputs are available - if not, create mock data for testing

        if not PYDECISION_AVAILABLE:
            print("\n[ERROR] PYDECISION NOT AVAILABLE")
            self.logger.error("pyDecision is not available. Cannot perform MCDA analysis")
            # Create default outputs to avoid validation errors
            print("   Creating default empty outputs...")
            mcda_rankings = pd.DataFrame()
            ranking_scores = pd.DataFrame()
            correlation_matrix = pd.DataFrame()
            best_alternatives = pd.DataFrame()
            analysis_summary = {'error': 'pyDecision not available'}

            outputs_dict = {
                'mcda_rankings': mcda_rankings,
                'ranking_scores': ranking_scores,
                'correlation_matrix': correlation_matrix,
                'best_alternatives': best_alternatives,
                'analysis_summary': analysis_summary,
            }

            print("   Storing default outputs and returning...")
            self.store_sos_outputs_values(outputs_dict)
            print("[ERROR] VALUE ANALYSIS FAILED - PYDECISION NOT AVAILABLE")
            print("="*80)
            return

        # Convert DoE dictionary outputs to a combined dataframe
        # Dictionaries are scenario_name -> value mappings
        print("\n[PROCESSING] DATA PROCESSING")
        print("   Converting dictionaries to DataFrame...")


        samples_output_df = self.get_sosdisc_inputs('samples_outputs_df')
        samples_inputs_df = self.get_sosdisc_inputs('samples_inputs_df')



        # Auto-detect criteria names if not provided
        print("\n[ANALYSIS] CRITERIA DETECTION")
        if not input_criteria_names and not output_criteria_names:
            print("   Auto-detecting criteria names...")
            # Assume DoE input variables are criteria (continuous parameters)
            # Exclude 'scenario' column which contains string identifiers
            input_criteria_names = [col for col in samples_inputs_df.columns if col not in ['scenario_name']]
            output_criteria_names = [col for col in samples_output_df.columns if col not in ['scenario_name']]
            print(f"   - Auto-detected input criteria: {input_criteria_names}")
            print(f"   - Auto-detected output criteria: {output_criteria_names}")
            self.logger.info(f"Auto-detected input criteria: {input_criteria_names}")
            self.logger.info(f"Auto-detected output criteria: {output_criteria_names}")
        else:
            print("   Using provided criteria names:")
            print(f"   - Input criteria: {input_criteria_names}")
            print(f"   - Output criteria: {output_criteria_names}")

        # Combine input and output criteria names
        all_criteria_names =  output_criteria_names
        # Prepare decision matrix
        print("\n[PROCESSING] DECISION MATRIX PREPARATION")
        print(f"   Extracting columns: {all_criteria_names}")
        #decision_matrix = pd.merge(samples_inputs_df,samples_output_df,on='scenario_name',how='outer')[all_criteria_names].values
        decision_matrix= samples_output_df[all_criteria_names].values
        print(f"   - Matrix shape: {decision_matrix.shape}")
        print(f"   - Data type: {decision_matrix.dtype}")

        # Ensure decision matrix contains only numeric values

        print("   Converting to float...")
        decision_matrix = decision_matrix.astype(float)
        print("[OK] Decision matrix converted successfully")
        print(f"   - Final shape: {decision_matrix.shape}")
        print(f"   - Final dtype: {decision_matrix.dtype}")

        self.logger.info(f"Decision matrix converted to float, shape: {decision_matrix.shape}")
        self.logger.info(f"Decision matrix sample:\n{decision_matrix[:3]}")


        criteria_weights = np.array(criteria_weights)
        print(f"   - Final weights array: {criteria_weights}")

        # Set default criteria types if not provided
        print("\n[SETUP] CRITERIA TYPES SETUP")
        print(f"   Input types: {criteria_types} (type: {type(criteria_types)})")

        if not criteria_types or len(criteria_types) != len(all_criteria_names):
            criteria_types = ['max'] * len(all_criteria_names)
            print(f"   Using default 'max' for all: {criteria_types}")
            self.logger.info(f"Using 'max' type for all criteria: {criteria_types}")
        else:
            print(f"   Using provided types: {criteria_types}")

        # Debug log the criteria_types type and value
        self.logger.info(f"Debug: criteria_types type: {type(criteria_types)}, value: {criteria_types}")

        # Ensure criteria_types is a list of strings
        if isinstance(criteria_types, str):
            print("   [WARN] WARNING: criteria_types is a string, converting to default")
            self.logger.error(f"criteria_types is a string: {criteria_types}")
            criteria_types = ['max'] * len(all_criteria_names)
        elif not isinstance(criteria_types, (list, tuple)):
            print("   [WARN] WARNING: criteria_types is not a list, converting to default")
            self.logger.error(f"criteria_types is not a list: {type(criteria_types)}")
            criteria_types = ['max'] * len(all_criteria_names)

        # Convert criteria types to numerical format for pyDecision
        # 'max' = 1 (higher is better), 'min' = -1 (lower is better)
        criteria_types_numerical = []
        for ctype in criteria_types:
            if ctype.lower() == 'max':
                criteria_types_numerical.append(1)
            elif ctype.lower() == 'min':
                criteria_types_numerical.append(-1)
            else:
                criteria_types_numerical.append(1)  # Default to max

        criteria_types_numerical = np.array(criteria_types_numerical)
        print(f"   - Final types array: {criteria_types_numerical} (1=max, -1=min)")

        # Apply MCDA methods
        print("\n[MCDA] MCDA METHODS APPLICATION")
        print(f"   Methods to apply: {mcda_methods}")

        mcda_methods = ['topsis', 'saw']
        rankings_dict = {}
        scores_dict = {}

        print("   Starting MCDA analysis with:")
        print(f"   - Decision matrix shape: {decision_matrix.shape}")
        print(f"   - Criteria weights: {criteria_weights}")
        print(f"   - Criteria types: {criteria_types_numerical}")

        self.logger.info("Starting MCDA methods application")
        self.logger.info(f"criteria_weights: {criteria_weights} (type: {type(criteria_weights)})")
        self.logger.info(f"decision_matrix shape: {np.array(decision_matrix).shape}")
        self.logger.info(f"criteria_types_numerical: {criteria_types_numerical}")

        for method in mcda_methods:
            try:
                print(f"\n   [PROCESS] Applying {method.upper()} method...")
                self.logger.info(f"Trying method: {method}")

                if method.lower() == 'topsis':
                    print("      Calling topsis_method...")
                    result = topsis_method(decision_matrix, criteria_weights, criteria_types_numerical,
                                         graph=False, verbose=False)
                    print(f"      [OK] TOPSIS completed - Result type: {type(result)}")
                    print(f"      [OK] TOPSIS result shape: {result.shape if hasattr(result, 'shape') else 'no shape'}")
                    print(f"      [OK] TOPSIS sample values: {result[:3] if hasattr(result, '__len__') else result}")
                    self.logger.info(f"TOPSIS result: {result}")

                elif method.lower() == 'saw':
                    print("      Calling saw_method...")
                    result = saw_method(decision_matrix, criteria_weights, criteria_types_numerical,
                                      graph=False, verbose=False)
                    print(f"      [OK] SAW completed - Result type: {type(result)}")
                    print(f"      [OK] SAW result shape: {result.shape if hasattr(result, 'shape') else 'no shape'}")
                    print(f"      [OK] SAW sample values: {result[:3] if hasattr(result, '__len__') else result}")
                    self.logger.info(f"SAW result: {result}")

                # Process result
                print(f"      [PROCESS] Processing {method} results...")
                if result is not None:
                    if hasattr(result, 'ndim') and result.ndim == 2:
                        # Multi-dimensional result
                        print(f"         Multi-dimensional result: {result.shape}")
                        rankings_dict[f'{method}_rank'] = result[:, 1] if result.shape[1] > 1 else list(range(1, len(result) + 1))
                        scores_dict[f'{method}_score'] = result[:, 0]
                    elif hasattr(result, '__len__'):
                        # 1D array or list
                        print(f"         1D result with {len(result)} values")
                        if method.lower() == 'saw':
                            # SAW returns scores, calculate rankings
                            print("         Converting SAW scores to rankings...")
                            sorted_indices = np.argsort(-result)
                            rankings = np.empty_like(sorted_indices)
                            rankings[sorted_indices] = np.arange(1, len(result) + 1)
                            rankings_dict[f'{method}_rank'] = rankings
                            scores_dict[f'{method}_score'] = result
                        else:
                            rankings_dict[f'{method}_rank'] = list(range(1, len(result) + 1))
                            scores_dict[f'{method}_score'] = result
                    else:
                        # Single value or unexpected format
                        print(f"         Unexpected result format: {type(result)}")
                        rankings_dict[f'{method}_rank'] = list(range(1, len(decision_matrix) + 1))
                        scores_dict[f'{method}_score'] = [0.0] * len(decision_matrix)
                else:
                    # No result
                    print("         No result returned")
                    rankings_dict[f'{method}_rank'] = list(range(1, len(decision_matrix) + 1))
                    scores_dict[f'{method}_score'] = [0.0] * len(decision_matrix)

                print(f"      [OK] {method.upper()} processing complete")

            except Exception as e:
                print(f"      [ERROR] ERROR with {method}: {str(e)}")
                print(f"         Error type: {type(e).__name__}")
                self.logger.error(f"Error with {method} method: {str(e)}")
                # Provide fallback values
                rankings_dict[f'{method}_rank'] = list(range(1, len(decision_matrix) + 1))
                scores_dict[f'{method}_score'] = [0.0] * len(decision_matrix)

        print("\n[RESULTS] MCDA RESULTS SUMMARY:")
        print(f"   - Methods completed: {list(rankings_dict.keys())}")
        print(f"   - Rankings available: {len(rankings_dict)} methods")
        print(f"   - Scores available: {len(scores_dict)} methods")

        # Create output dataframes
        print("\n[OUTPUT] CREATING OUTPUT DATAFRAMES")
        mcda_rankings = pd.DataFrame(rankings_dict, index=samples_output_df.index)
        ranking_scores = pd.DataFrame(scores_dict, index=samples_output_df.index)
        print(f"   - Rankings DataFrame shape: {mcda_rankings.shape}")
        print(f"   - Scores DataFrame shape: {ranking_scores.shape}")

        # Calculate correlation matrix if multiple methods
        correlation_matrix = pd.DataFrame()
        if len(rankings_dict) > 1:
            correlation_matrix = mcda_rankings.corr()
            print(f"   - Correlation matrix shape: {correlation_matrix.shape}")
        else:
            print(f"   - Skipping correlation matrix (only {len(rankings_dict)} method)")

        # Identify best alternatives (top 5 from each method)
        print("\n[BEST] IDENTIFYING BEST ALTERNATIVES")
        best_alternatives_dict = {}
        for method, rankings in rankings_dict.items():
            top_indices = np.argsort(rankings)[:5]  # Top 5
            print(f"   - {method} top 5 indices: {top_indices}")
            best_alternatives_dict[method] = {
                'indices': top_indices.tolist(),
                'values': [samples_output_df.iloc[i].to_dict() for i in top_indices]
            }

        best_alternatives = pd.DataFrame({
            method: [f"Alt_{i}" for i in data['indices'][:5]]
            for method, data in best_alternatives_dict.items()
        })
        print(f"   - Best alternatives DataFrame shape: {best_alternatives.shape}")

        # Analysis summary
        analysis_summary = {
            'total_alternatives': len(decision_matrix),
            'total_criteria': len(all_criteria_names),
            'methods_applied': list(rankings_dict.keys()),
            'criteria_names': all_criteria_names,
            'criteria_types': criteria_types,
            'criteria_weights': criteria_weights.tolist(),
        }
        print("\n[SUMMARY] ANALYSIS SUMMARY:")
        print(f"   - Total alternatives: {analysis_summary['total_alternatives']}")
        print(f"   - Total criteria: {analysis_summary['total_criteria']}")
        print(f"   - Methods applied: {analysis_summary['methods_applied']}")
        print(f"   - Criteria: {analysis_summary['criteria_names']}")

        # Store outputs
        print("\n[STORAGE] STORING OUTPUTS")
        dict_values = {
            'mcda_rankings': mcda_rankings,
            'ranking_scores': ranking_scores,
            'correlation_matrix': correlation_matrix,
            'best_alternatives': best_alternatives,
            'analysis_summary': analysis_summary,
        }

        print(f"   - Outputs to store: {list(dict_values.keys())}")
        for key, value in dict_values.items():
            if hasattr(value, 'shape'):
                print(f"     * {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"     * {key}: dict with {len(value)} keys")
            else:
                print(f"     * {key}: {type(value)}")

        self.store_sos_outputs_values(dict_values)

        print("\n[OK] VALUE ANALYSIS EXECUTION COMPLETED SUCCESSFULLY")
        print("="*80)

    def get_chart_filter_list(self):
        """Return chart filter list"""
        chart_filters = []
        chart_list = [
            'MCDA Rankings Comparison',
            'MCDA Scores',
            'Performance Scatter Plot',
            'Criteria Weights',
            'Best Alternatives',
            'Correlation Heatmap',
            'GAIA Analysis'
        ]

        chart_filters.append(ChartFilter('charts', chart_list, chart_list, 'charts'))
        return chart_filters

    def get_post_processing_list(self, chart_filters=None):
        """Generate post-processing charts mixing standard SoSTrades charts with custom Plotly visualizations"""
        print("\n[CHARTS] GENERATING POST-PROCESSING CHARTS")
        charts = []

        # Default charts if no filter
        if chart_filters is None:
            chart_list = [
                'MCDA Rankings Comparison',
                'MCDA Scores',
                'Performance Scatter Plot',
                'Criteria Weights',
                'Best Alternatives',
                'Correlation Heatmap',
                'GAIA Analysis'
            ]
        else:
            chart_list = []
            for chart_filter in chart_filters:
                if chart_filter.filter_key == 'charts':
                    chart_list = chart_filter.selected_values

        print(f"   Requested charts: {chart_list}")

        # Get output data
        try:
            mcda_rankings = self.get_sosdisc_outputs('mcda_rankings')
            ranking_scores = self.get_sosdisc_outputs('ranking_scores')
            correlation_matrix = self.get_sosdisc_outputs('correlation_matrix')
            best_alternatives = self.get_sosdisc_outputs('best_alternatives')
            analysis_summary = self.get_sosdisc_outputs('analysis_summary')

            print(f"   - Retrieved rankings: {mcda_rankings.shape if hasattr(mcda_rankings, 'shape') else type(mcda_rankings)}")
            print(f"   - Retrieved scores: {ranking_scores.shape if hasattr(ranking_scores, 'shape') else type(ranking_scores)}")
            print(f"   - Retrieved correlation: {correlation_matrix.shape if hasattr(correlation_matrix, 'shape') else type(correlation_matrix)}")

        except Exception as e:
            print(f"   [ERROR] Error retrieving outputs: {e}")
            self.logger.error(f"Error retrieving outputs for post-processing: {e}")
            return charts

        # 1. Standard SoSTrades Chart: MCDA Rankings Comparison
        if 'MCDA Rankings Comparison' in chart_list and hasattr(mcda_rankings, 'shape') and not mcda_rankings.empty:
            try:
                print("   [PROCESS] Creating MCDA Rankings Comparison chart...")
                chart = TwoAxesInstanciatedChart(
                    'Alternative', 'Ranking',
                    chart_name='MCDA Rankings Comparison'
                )

                alternatives = [f"Alt_{i+1}" for i in range(len(mcda_rankings))]

                for method_col in mcda_rankings.columns:
                    rankings = mcda_rankings[method_col].tolist()
                    series = InstanciatedSeries(
                        alternatives, rankings,
                        method_col, 'lines'
                    )
                    chart.series.append(series)

                charts.append(chart)
                print("   [OK] MCDA Rankings Comparison chart created")

            except Exception as e:
                print(f"   [ERROR] Error creating Rankings chart: {e}")
                self.logger.error(f"Error creating MCDA Rankings chart: {e}")

        # 2. Standard SoSTrades Chart: MCDA Scores
        if 'MCDA Scores' in chart_list and hasattr(ranking_scores, 'shape') and not ranking_scores.empty:
            try:
                print("   [PROCESS] Creating MCDA Scores chart...")
                chart = TwoAxesInstanciatedChart(
                    'Alternative', 'Score',
                    chart_name='MCDA Method Scores'
                )

                alternatives = [f"Alt_{i+1}" for i in range(len(ranking_scores))]

                for method_col in ranking_scores.columns:
                    scores = ranking_scores[method_col].tolist()
                    series = InstanciatedSeries(
                        alternatives, scores,
                        method_col, 'bar'
                    )
                    chart.series.append(series)

                charts.append(chart)
                print("   [OK] MCDA Scores chart created")

            except Exception as e:
                print(f"   [ERROR] Error creating Scores chart: {e}")
                self.logger.error(f"Error creating MCDA Scores chart: {e}")

        # 3. Standard SoSTrades Chart: Criteria Weights
        if 'Criteria Weights' in chart_list and analysis_summary and isinstance(analysis_summary, dict):
            try:
                print("   [PROCESS] Creating Criteria Weights chart...")
                criteria_names = analysis_summary.get('criteria_names', [])
                criteria_weights = analysis_summary.get('criteria_weights', [])

                if criteria_names and criteria_weights:
                    chart = TwoAxesInstanciatedChart(
                        'Criteria', 'Weight',
                        chart_name='Criteria Weights'
                    )

                    series = InstanciatedSeries(
                        criteria_names, criteria_weights,
                        'Weights', 'bar'
                    )
                    chart.series.append(series)
                    charts.append(chart)
                    print("   [OK] Criteria Weights chart created")

            except Exception as e:
                print(f"   [ERROR] Error creating Criteria Weights chart: {e}")
                self.logger.error(f"Error creating Criteria Weights chart: {e}")

        # 4. Custom Plotly Chart: Performance Scatter Plot
        if 'Performance Scatter Plot' in chart_list and PLOTLY_AVAILABLE:
            try:
                print("   [PROCESS] Creating Performance Scatter Plot...")
                scatter_chart = self._create_performance_scatter_plot()
                if scatter_chart:
                    charts.append(scatter_chart)
                    print("   [OK] Performance Scatter Plot created")
                else:
                    print("   [WARN] Performance Scatter Plot creation returned None")

            except Exception as e:
                print(f"   [ERROR] Error creating Performance Scatter: {e}")
                self.logger.error(f"Error creating Performance Scatter Plot: {e}")

        # 5. Custom Plotly Chart: Correlation Heatmap
        if 'Correlation Heatmap' in chart_list and PLOTLY_AVAILABLE and hasattr(correlation_matrix, 'shape') and not correlation_matrix.empty:
            try:
                print("   [PROCESS] Creating Correlation Heatmap...")
                heatmap_chart = self._create_correlation_heatmap(correlation_matrix)
                if heatmap_chart:
                    charts.append(heatmap_chart)
                    print("   [OK] Correlation Heatmap created")
                else:
                    print("   [WARN] Correlation Heatmap creation returned None")

            except Exception as e:
                print(f"   [ERROR] Error creating Correlation Heatmap: {e}")
                self.logger.error(f"Error creating Correlation Heatmap: {e}")

        # 6. Custom Plotly Chart: GAIA Analysis (if available)
        if 'GAIA Analysis' in chart_list and PLOTLY_AVAILABLE and SKLEARN_AVAILABLE:
            try:
                print("   [PROCESS] Creating GAIA Analysis...")
                gaia_chart = self._create_gaia_analysis()
                if gaia_chart:
                    charts.append(gaia_chart)
                    print("   [OK] GAIA Analysis created")
                else:
                    print("   [WARN] GAIA Analysis creation returned None")

            except Exception as e:
                print(f"   [ERROR] Error creating GAIA Analysis: {e}")
                self.logger.error(f"Error creating GAIA Analysis: {e}")

        # Handle unavailable dependencies
        if not PLOTLY_AVAILABLE and any(chart in chart_list for chart in ['Performance Scatter Plot', 'Correlation Heatmap', 'GAIA Analysis']):
            print("   [WARN] Some charts require Plotly but it's not available")

        if not SKLEARN_AVAILABLE and 'GAIA Analysis' in chart_list:
            print("   [WARN] GAIA Analysis requires scikit-learn but it's not available")

        print(f"   [CHARTS] Total charts generated: {len(charts)}")
        return charts

    def _create_performance_scatter_plot(self):
        """Create an interactive scatter plot showing alternative performance"""
        try:
            # Get current outputs
            analysis_summary = self.get_sosdisc_outputs('analysis_summary')
            ranking_scores = self.get_sosdisc_outputs('ranking_scores')

            if not analysis_summary or not hasattr(ranking_scores, 'shape') or ranking_scores.empty:
                print("      No data available for Performance Scatter Plot")
                return None

            criteria_names = analysis_summary.get('criteria_names', [])
            if len(criteria_names) < 2:
                print("      Need at least 2 criteria for scatter plot")
                return None

            # Get DoE data for scatter plot axes
            top_speed_dict = self.get_sosdisc_inputs('top_speed_dict')
            range_dict = self.get_sosdisc_inputs('range_dict')
            manufacturing_cost_dict = self.get_sosdisc_inputs('manufacturing_cost_dict')

            if not all([top_speed_dict, range_dict, manufacturing_cost_dict]):
                print("      DoE data not available for scatter plot")
                return None

            # Convert to lists for plotting
            scenarios = list(top_speed_dict.keys())
            top_speeds = [top_speed_dict[s] for s in scenarios]
            ranges = [range_dict[s] for s in scenarios]
            costs = [manufacturing_cost_dict[s] for s in scenarios]

            # Get TOPSIS scores if available for color coding
            color_values = None
            if 'topsis_score' in ranking_scores.columns:
                color_values = ranking_scores['topsis_score'].tolist()

            # Create scatter plot
            fig = go.Figure()

            # Main scatter plot
            scatter_trace = go.Scatter(
                x=top_speeds,
                y=ranges,
                mode='markers',
                marker=dict(
                    size=12,
                    color=color_values if color_values else costs,
                    colorscale='Viridis',
                    colorbar=dict(
                        title="TOPSIS Score" if color_values else "Cost (EUR)"
                    ),
                    line=dict(width=1, color='white')
                ),
                text=[f"Scenario: {s}<br>Top Speed: {ts:.1f} km/h<br>Range: {r:.1f} km<br>Cost: {c:.0f} EUR"
                      for s, ts, r, c in zip(scenarios, top_speeds, ranges, costs)],
                hovertemplate='%{text}<extra></extra>',
                name='Alternatives'
            )
            fig.add_trace(scatter_trace)

            # Update layout
            fig.update_layout(
                title='Alternative Performance Analysis',
                xaxis_title='Top Speed (km/h)',
                yaxis_title='Range (km)',
                width=800,
                height=600,
                template='plotly_white'
            )

            return ValueAnalysisChart(fig, "Performance Scatter Plot")

        except Exception as e:
            print(f"      Error creating Performance Scatter Plot: {e}")
            self.logger.error(f"Error in _create_performance_scatter_plot: {e}")
            return None

    def _create_correlation_heatmap(self, correlation_matrix):
        """Create a correlation heatmap between MCDA methods"""
        try:
            if correlation_matrix.empty:
                return None

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 14},
                hoverongaps=False
            ))

            fig.update_layout(
                title='MCDA Method Correlation Matrix',
                width=600,
                height=500,
                template='plotly_white'
            )

            return ValueAnalysisChart(fig, "Correlation Heatmap")

        except Exception as e:
            print(f"      Error creating Correlation Heatmap: {e}")
            self.logger.error(f"Error in _create_correlation_heatmap: {e}")
            return None

    def _create_gaia_analysis(self):
        """Create GAIA-style PCA visualization if sklearn is available"""
        try:
            if not SKLEARN_AVAILABLE:
                print("      Scikit-learn not available for GAIA analysis")
                return None

            # Get decision matrix data
            top_speed_dict = self.get_sosdisc_inputs('top_speed_dict')
            range_dict = self.get_sosdisc_inputs('range_dict')
            manufacturing_cost_dict = self.get_sosdisc_inputs('manufacturing_cost_dict')

            if not all([top_speed_dict, range_dict, manufacturing_cost_dict]):
                print("      DoE data not available for GAIA analysis")
                return None

            # Create decision matrix
            scenarios = list(top_speed_dict.keys())
            decision_matrix = np.array([
                [top_speed_dict[s] for s in scenarios],
                [range_dict[s] for s in scenarios],
                [manufacturing_cost_dict[s] for s in scenarios]
            ]).T

            # Standardize data
            scaler = StandardScaler()
            standardized_data = scaler.fit_transform(decision_matrix)

            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(standardized_data)

            # Get criteria info
            analysis_summary = self.get_sosdisc_outputs('analysis_summary')
            criteria_names = analysis_summary.get('criteria_names', ['top_speed', 'range', 'manufacturing_cost'])

            # Create scatter plot
            fig = go.Figure()

            # Plot alternatives
            fig.add_trace(go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode='markers+text',
                marker=dict(size=10, color='blue'),
                text=[f"A{i+1}" for i in range(len(scenarios))],
                textposition="top center",
                name='Alternatives',
                hovertemplate='Alternative %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))

            # Plot criteria vectors (loadings)
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            for i, criterion in enumerate(criteria_names):
                fig.add_trace(go.Scatter(
                    x=[0, loadings[i, 0]],
                    y=[0, loadings[i, 1]],
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=[0, 8]),
                    name=criterion,
                    showlegend=True
                ))

                # Add criterion labels
                fig.add_annotation(
                    x=loadings[i, 0] * 1.1,
                    y=loadings[i, 1] * 1.1,
                    text=criterion,
                    showarrow=False,
                    font=dict(color='red', size=12)
                )

            # Update layout
            fig.update_layout(
                title=f'GAIA Analysis (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})',
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                width=800,
                height=700,
                template='plotly_white'
            )

            return ValueAnalysisChart(fig, "GAIA Analysis")

        except Exception as e:
            print(f"      Error creating GAIA Analysis: {e}")
            self.logger.error(f"Error in _create_gaia_analysis: {e}")
            return None
# Export the main discipline class for SoSTrades factory
__all__ = ['ValueAnalysisDiscipline']
