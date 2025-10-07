# ValueAnalysis Discipline

This discipline provides multi-criteria decision analysis (MCDA) capabilities for analyzing Design of Experiments (DoE) results using the pyDecision library with Plotly visualizations.

## Features

- **Multi-Criteria Decision Analysis**: Supports multiple MCDA methods including TOPSIS, SAW, VIKOR, and PROMETHEE GAIA
- **GAIA Visualization**: Interactive GAIA charts using Plotly instead of matplotlib
- **Correlation Analysis**: Correlation matrix between different MCDA methods
- **Ranking Comparison**: Visual comparison of rankings across methods
- **DoE Integration**: Seamless integration with SoSTrades DoE generation

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_value_analysis.txt
```

## Usage

### In a SoSTrades Process

The discipline can be used in a process that includes:
1. DoE generation (SampleGenerator)
2. Multi-criteria model evaluation
3. ValueAnalysis for decision analysis

### Inputs

- `doe_results`: DataFrame containing DoE results with input variables and output criteria
- `input_criteria_names`: List of input criteria/variable names
- `output_criteria_names`: List of output criteria/objective names  
- `criteria_types`: List specifying "max" for benefit criteria, "min" for cost criteria
- `criteria_weights`: Optional weights for each criterion
- `mcda_methods`: MCDA methods to apply (default: ['topsis', 'saw', 'vikor'])
- `generate_gaia_chart`: Boolean to generate GAIA visualization
- `generate_correlation_chart`: Boolean to generate correlation matrix
- `generate_ranking_comparison`: Boolean to generate ranking comparison

### Outputs

- `mcda_rankings`: Rankings from different MCDA methods
- `ranking_scores`: Scores from different MCDA methods
- `correlation_matrix`: Correlation matrix between ranking methods
- `best_alternatives`: Top alternatives identified by each method
- `analysis_summary`: Summary of the multi-criteria analysis

### Visualizations

1. **GAIA Chart**: Principal Component Analysis visualization showing alternatives and criteria in a 2D space
2. **Correlation Matrix**: Heatmap showing correlations between different MCDA methods
3. **Ranking Comparison**: Line plot comparing rankings across methods

## Example

See `usecase_value_analysis_doe.py` for a complete example that includes:
- DoE generation with Latin Hypercube Sampling
- Dummy multi-criteria model with 6 output criteria
- ValueAnalysis with multiple MCDA methods
- Interactive Plotly charts

## Test Process

The test process includes:
- **process.py**: Process builder combining DoE, dummy model, and ValueAnalysis
- **dummy_multi_criteria_model.py**: Example multi-criteria model with realistic trade-offs
- **usecase_value_analysis_doe.py**: Complete usecase example
- **test_value_analysis.py**: Unit tests

## Dependencies

- `pyDecision`: For MCDA algorithms
- `plotly`: For interactive visualizations
- `scikit-learn`: For PCA in GAIA charts (optional)
- `pandas`, `numpy`: For data manipulation
- Standard SoSTrades dependencies

## Notes

- The discipline gracefully handles missing dependencies
- Falls back to simpler visualizations if scikit-learn is not available
- Supports custom MCDA methods through the extensible architecture
- All charts are generated using Plotly for better interactivity compared to matplotlib

## Future Enhancements

- Integration with additional pyDecision methods
- Support for fuzzy MCDA methods
- Advanced GAIA features (decision stick, compromise direction)
- Export capabilities for rankings and visualizations
