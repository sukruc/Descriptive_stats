## Descriptive Statistics

### Description:
This module contains a helper/utility function that can visualize descriptive
statistics of a timeseries variable.

It can also analyze for outliers and mark inconsistent and potentially erroneous
data based on 3 different methods:
* Mean and standard deviation,
* Median and median absolute deviation,
* Median and interquartile deviation.

Those methods above are also applied to gradients to identify inconsistent data points.
### Usage:
`DescriptiveStatistics` works on series. Outputs are a csv file with summary and
a pdf file containing 6 plots:
different plots:
* Descriptive statistics as text,
* a histogram,
* a boxplot,
* a lineplot with outlier thresholds,
* a lineplot of gradients with outlier thresholds,
* and a scatterplot of clean data.

Please refer to `DescriptiveStatistics_example.py` for usage details and examples.
