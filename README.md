# Assignment 2 – Cyclic Cross Correlation

In this assignment, we delve into the world of cyclic cross-correlation, a fundamental tool for detecting similarities between discrete signals. Given signals $x(k)$
and  $y(k)$, where $0 \leq k < N$, the circular cross-correlation ( $rxy (\tau)$ ) is defined by the formula:

$rxy (\tau) = \sum_{k=0}^{n-1} x(k) \cdot y[(\tau+k) \mod n]$

The reference implementation, housed in `cyclicCircConv.cu`, facilitates exploration by running the computation kernel on both the CPU and GPU. Our primary task is to optimize the thread launch grid and assess the viability of offloading computations to a GPU device. Using `cyclicCircConvDouble.cu`, we aim to present execution time plots and explanations for various thread launch grids, while also investigating factors influencing GPU performance. The ultimate goal is to draw conclusions regarding the efficiency of GPU offloading.

## Contributors

| Author                  | N Mec | Contribution Percentage |
|-------------------------|------------|-------------------------|
| Gonçalo Silva         | 103244      | 50%                     |
| Catarina Barroqueiro    | 103895     | 50%                     |




## Summary: Significant Figures

The general rule establishes that the number of significant figures in the result should not exceed the number from the most imprecise measurement used.

### Provided Data:
- Mean Value (Average): 2.538921 (7 significant figures)
- Standard Deviation: 0.0025 (4 significant figures)

### Application of the Rule:
By rounding the mean value to match the number of significant figures in the standard deviation, we obtain:
- Rounded Mean Value: 2.539 (4 significant figures)

Therefore, to maintain consistent precision, the final result should be expressed as 2.539, with four significant figures.
