"""
Post-processing charts for ValueAnalysis discipline

Display pyDecision plotly graphs in SoSTrades
(not using SoSTrades predefined graphs but directly as fior the map in LogisticsNetworkOptimizer)
"""

from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


def post_processing_filters(execution_engine, namespace):
    """Return chart filters for ValueAnalysis"""
    chart_filters = []
    chart_filters.append(ChartFilter('MCDA Rankings', True, 'mcda_rankings'))
    chart_filters.append(ChartFilter('Performance Scores', True, 'performance_scores'))
    chart_filters.append(ChartFilter('Best Alternatives', True, 'best_alternatives'))

    return chart_filters

def post_processings(execution_engine, namespace, chart_filters):
    """Generate post-processing charts for ValueAnalysis"""
    instanciated_charts = []

    try:
        # Get MCDA results
        mcda_rankings = execution_engine.dm.get_value(f'{namespace}.mcda_rankings')
        ranking_scores = execution_engine.dm.get_value(f'{namespace}.ranking_scores')
        analysis_summary = execution_engine.dm.get_value(f'{namespace}.analysis_summary')

        # Get selected filters - handle both list and None cases
        selected_filters = []
        if chart_filters:
            selected_filters = [f.filter_key for f in chart_filters if hasattr(f, 'selected') and f.selected]

        # If no specific filters selected, show all charts
        if not selected_filters:
            selected_filters = ['mcda_rankings', 'performance_scores', 'best_alternatives']

        # Only create charts if we have the required data
        if (mcda_rankings is not None and
            not mcda_rankings.empty and
            ranking_scores is not None and
            not ranking_scores.empty):

            # Chart 1: MCDA Rankings Comparison
            if 'mcda_rankings' in selected_filters:
                ranking_chart = TwoAxesInstanciatedChart('Alternative', 'Ranking',
                                                       'MCDA Rankings Comparison')

                alternatives = [f'Alt_{i+1}' for i in mcda_rankings.index]

                for method in mcda_rankings.columns:
                    rankings = [float(v) for v in mcda_rankings[method].values]
                    new_serie = InstanciatedSeries(list(alternatives),
                                                   list(rankings),
                                                   f'{method.upper()} Rankings', 'lines+markers')

                    ranking_chart.add_series(new_serie)

                instanciated_charts.append(ranking_chart)

            # Chart 2: Performance Scores
            if 'performance_scores' in selected_filters:
                scores_chart = TwoAxesInstanciatedChart('Alternative', 'Score',
                                                      'MCDA Performance Scores')

                alternatives = [f'Alt_{i+1}' for i in ranking_scores.index]

                for method in ranking_scores.columns:
                    scores = [float(v) for v in ranking_scores[method].values]
                    new_serie = InstanciatedSeries(list(alternatives),
                                                   list(scores),
                                                   f'{method.upper()} Scores', 'bar')

                    scores_chart.add_series(new_serie)

                instanciated_charts.append(scores_chart)

            # Chart 3: Best Alternatives Summary
            if 'best_alternatives' in selected_filters and analysis_summary:
                summary_chart = TwoAxesInstanciatedChart('Method', 'Best Alternative Index',
                                                        'Best Alternative by Method')

                if isinstance(analysis_summary, dict) and 'methods_applied' in analysis_summary:
                    methods = analysis_summary['methods_applied']
                    best_indices = []

                    for method in methods:
                        if method in ranking_scores.columns:
                            # Find index of best (highest) score
                            best_idx = ranking_scores[method].idxmax()
                            best_indices.append(int(best_idx))
                        else:
                            best_indices.append(0)
                    new_serie = InstanciatedSeries(list(methods),
                                                   list(best_indices),
                                                   'Best Alternative Index', 'bar')

                    summary_chart.add_series(new_serie)
                    instanciated_charts.append(summary_chart)

    except Exception as e:
        # Log error but don't fail - return empty list
        print(f"Error in ValueAnalysis post-processing: {e}")

    return instanciated_charts
