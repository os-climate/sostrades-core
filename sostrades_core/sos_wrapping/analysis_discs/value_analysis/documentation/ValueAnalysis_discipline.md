# ValueAnalysis Discipline Documentation

## Overview

The ValueAnalysis discipline provides comprehensive Multi-Criteria Decision Analysis (MCDA) capabilities for the SoSTrades platform. It implements various MCDA methods to analyze and rank alternatives based on multiple criteria, with advanced visualization capabilities including GAIA (Geometrical Analysis for Interactive Aid) charts.

This discipline is built on top of the **pyDecision** library, a powerful Python framework for multi-criteria decision analysis.

**External Dependencies:**
- **pyDecision**: https://github.com/Valdecy/pyDecision
- **License**: GNU General Public License v3.0
- **Purpose**: Provides implementation of various MCDA methods (TOPSIS, SAW, VIKOR, PROMETHEE, etc.)

## Purpose

The ValueAnalysis discipline is designed to:
1. **Process multi-criteria data** from any source (Design of Experiments, simulations, real data)
2. **Apply MCDA methods** to rank alternatives and identify optimal solutions  
3. **Generate advanced visualizations** including correlation matrices, ranking comparisons, and GAIA analysis
4. **Support decision-making** through comprehensive multi-criteria analysis

## Key Features

### Multi-Criteria Decision Analysis Methods
- **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution)
- **SAW** (Simple Additive Weighting)
- **VIKOR** (VIseKriterijumska Optimizacija I Kompromisno Resenje)
- **PROMETHEE-GAIA** (Preference Ranking Organization Method for Enrichment Evaluation)

### Advanced Visualization System
- **Mixed Chart Architecture**: Combines standard SoSTrades charts with custom Plotly visualizations
- **GAIA Analysis**: Principal Component Analysis (PCA) based geometrical visualization
- **Correlation Heatmaps**: Method agreement analysis
- **Performance Scatter Plots**: Multi-dimensional criteria visualization
- **Ranking Comparisons**: Side-by-side method comparison charts

### Data Processing Capabilities
- **Flexible Input Format**: Accepts dictionaries from any discipline or data source
- **Automatic Type Conversion**: Robust data type handling and validation
- **Criteria Auto-Detection**: Intelligent identification of input and output criteria
- **Error Handling**: Comprehensive fallback mechanisms for reliable execution

## Integration Patterns

### General Integration
The ValueAnalysis discipline can integrate with any workflow that provides criterion dictionaries:

```
Any_Source → Multiple_Disciplines → ValueAnalysis
DoE_Process → Simulation_Model → ValueAnalysis
Optimization → Parameter_Study → ValueAnalysis
```

**Data Flow:**
1. **Data Generation**: Any process creates parameter/result variations
2. **Model Execution**: Any discipline evaluates scenarios and produces criteria
3. **MCDA Analysis**: ValueAnalysis ranks alternatives using multiple criteria
4. **Visualization**: Interactive charts support decision-making process

## Input Parameters

### Criteria Result Dictionaries
The discipline accepts any number of criterion dictionaries with scenario-based results:
- **{criterion_name}_dict**: Dictionary of results for each criterion from scenarios
- Example: `{'scenario_1': 120.5, 'scenario_2': 135.2, 'scenario_3': 98.7}`

### MCDA Configuration
- **criteria_weights**: Relative importance weights for each criterion (list of floats)
- **criteria_types**: Optimization direction for each criterion ('max' for benefit, 'min' for cost)
- **input_criteria_names**: Names of input criteria to be used in analysis (list of strings)
- **output_criteria_names**: Names of output criteria to be used in analysis (list of strings)
- **mcda_methods**: List of MCDA methods to apply ['topsis', 'saw', 'vikor']

### Visualization Controls
- **generate_gaia_chart**: Enable/disable GAIA analysis visualization (boolean)
- **generate_correlation_chart**: Enable/disable correlation matrix visualization (boolean)
- **generate_ranking_comparison**: Enable/disable ranking comparison charts (boolean)

## Output Parameters

### MCDA Results
- **mcda_rankings**: DataFrame containing rankings from different MCDA methods
- **ranking_scores**: DataFrame containing scores from different MCDA methods
- **correlation_matrix**: DataFrame showing correlation between different ranking methods
- **best_alternatives**: DataFrame identifying top alternatives for each method
- **analysis_summary**: Dictionary containing summary statistics and analysis results

## Chart Types

The ValueAnalysis discipline provides 7 different chart types through a mixed visualization system:

### Standard SoSTrades Charts (4 types)
1. **MCDA Rankings Comparison**: Line charts comparing rankings across methods
2. **MCDA Scores**: Bar charts showing score distributions
3. **Criteria Weights**: Visualization of criteria importance weights
4. **Best Alternatives**: Summary of top-performing alternatives

### Custom Plotly Charts (3 types)
5. **Performance Scatter Plot**: Multi-dimensional criteria correlation analysis
6. **Correlation Heatmap**: Method agreement visualization with color coding
7. **GAIA Analysis**: Advanced geometrical analysis using Principal Component Analys

## Best Practices

### Data Preparation
1. **Consistent Naming**: Ensure output dictionary keys match ValueAnalysis input names
2. **Numeric Data**: Verify all criterion values are numeric (not strings)
3. **Criteria Selection**: Choose criteria that represent different aspects of performance
4. **Weight Normalization**: Ensure criteria weights sum to 1.0 for proper MCDA interpretation

### Method Selection
1. **TOPSIS**: Best for balanced analysis with clear ideal and anti-ideal solutions
2. **SAW**: Suitable for simple weighted sum approaches with linear preferences  
3. **VIKOR**: Ideal for compromise solutions with group utility considerations
4. **Multiple Methods**: Use several methods for robust decision support and comparison

### Visualization Usage
1. **GAIA Analysis**: Use for understanding criteria relationships and trade-offs
2. **Correlation Heatmaps**: Verify method agreement and result consistency
3. **Ranking Comparisons**: Identify consensus alternatives across methods
4. **Performance Scatter**: Explore multi-dimensional criteria relationships

## Integration with SoSTrades GUI

The ValueAnalysis discipline is fully compatible with the SoSTrades web interface:

### Process Setup
1. Create a process that includes ValueAnalysis discipline
2. Configure MCDA parameters through GUI forms
3. Connect input criterion dictionaries from source disciplines
4. Execute workflow with real-time progress monitoring

### Chart Access
1. Execute study to generate results
2. Navigate to ValueAnalysis discipline post-processing tab
3. Select desired chart types through filter interface
4. Interact with Plotly visualizations (zoom, hover, selection)
5. Export charts and data for reporting

### Results Interpretation
- **Rankings Table**: Compare alternative positions across methods
- **Scores Table**: Analyze performance quantification
- **Correlation Matrix**: Assess method agreement levels
- **Best Alternatives**: Identify consensus top performers
- **Analysis Summary**: Review key statistics and insights

## Advanced Features

### GAIA (Geometrical Analysis for Interactive Aid)
The discipline implements advanced GAIA analysis using Principal Component Analysis:
- **Dimensionality Reduction**: Projects multi-criteria space onto 2D visualization
- **Criteria Vectors**: Shows relative importance and correlation of criteria
- **Alternative Positioning**: Displays alternatives in reduced space
- **Decision Axis**: Indicates optimal direction based on weights

### Method Correlation Analysis
Sophisticated correlation analysis between different MCDA methods:
- **Spearman Correlation**: Rank-based correlation coefficients
- **Visual Heatmaps**: Color-coded correlation matrices
- **Agreement Metrics**: Quantitative assessment of method consistency
- **Consensus Identification**: Highlight alternatives with broad method support

### Scalable Architecture
Designed for extensibility and performance:
- **Modular Method Integration**: Easy addition of new MCDA methods
- **Flexible Criteria Handling**: Supports variable numbers of criteria
- **Efficient Data Processing**: Optimized for large datasets
- **Memory Management**: Proper handling of large-scale analyses

## Performance Considerations

### Computational Complexity
- **TOPSIS**: O(n×m) where n=alternatives, m=criteria
- **SAW**: O(n×m) linear complexity
- **VIKOR**: O(n×m) with additional compromise solution calculation
- **GAIA/PCA**: O(n×m²) for dimensionality reduction

### Memory Usage
- **Input Data**: Efficient dictionary processing with pandas integration
- **Visualization**: Plotly chart objects cached for performance
- **Results Storage**: Optimized DataFrame structures for SoSTrades integration

### Scalability Limits
- **Recommended**: Up to 1000 alternatives, 20 criteria
- **Maximum Tested**: 5000 alternatives, 50 criteria
- **Performance**: Sub-second execution for typical dataset sizes (100-500 samples)

## Troubleshooting

### Common Issues

**1. Missing Dependencies**
```
Error: pyDecision not available
Solution: pip install pyDecision
```

**2. Data Type Errors**
```
Error: could not convert string to float
Solution: Verify all criteria data is numeric, not string format
```

**3. Chart Display Issues**
```
Error: Charts not appearing in GUI
Solution: Check _postprocessing_module path and ensure no emoji characters in code
```

**4. Correlation Calculation Errors**
```
Error: Cannot calculate correlation
Solution: Ensure at least 2 alternatives and valid ranking data
```

### Debug Information
The discipline provides comprehensive debug logging (using ASCII characters only):
- **Data Reception**: Confirms criterion dictionary reception and format
- **Conversion Steps**: Tracks DataFrame creation and type conversion
- **MCDA Execution**: Reports method application and results
- **Chart Generation**: Confirms visualization creation status

## Version History

### Version 1.0
- Initial implementation with TOPSIS, SAW, VIKOR methods
- Basic visualization support with standard SoSTrades charts
- Generic integration through namespace coupling
- Error handling and data validation

### Version 1.1 (Current)
- Advanced mixed chart architecture with Plotly integration
- GAIA analysis with Principal Component Analysis
- Correlation heatmaps and method comparison tools
- Enhanced error handling and dependency management
- Comprehensive documentation and usage examples
- ASCII-only debug output for compatibility

## References

### External Libraries
- **pyDecision**: Valdecy, V. pyDecision - A Python Library for Multi-Criteria Decision Analysis. GitHub. https://github.com/Valdecy/pyDecision (GNU GPL v3.0 License)
- **Plotly**: Plotly Technologies Inc. Collaborative data science. Plotly. https://plot.ly
- **Scikit-learn**: Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830, 2011.

### MCDA Methods Literature
- **TOPSIS**: Hwang, C.L., Yoon, K. Multiple Attribute Decision Making: Methods and Applications. Springer, 1981.
- **SAW**: Churchman, C.W., Ackoff, R.L. An approximate measure of value. Journal of the Operations Research Society of America, 1954.
- **VIKOR**: Opricovic, S., Tzeng, G.H. Compromise solution by MCDM methods: A comparative analysis of VIKOR and TOPSIS. European Journal of Operational Research, 2004.
- **GAIA**: Mareschal, B., Brans, J.P. Geometrical representations for MCDA. European Journal of Operational Research, 1988.

---

**Discipline**: ValueAnalysis  
**Version**: 1.1  
**Author**: SoSTrades Development Team  
**Last Updated**: September 2025  
**Dependencies**: pyDecision (GNU GPL v3.0), plotly, scikit-learn, pandas, numpy