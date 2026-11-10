# Restaurant tipping analysis

Hypothesis testing and simple linear regression on the seaborn `tips` dataset
(244 restaurant bills). Two questions: does tip amount differ by customer
gender, and how much of the tip is explained by the size of the bill?

## Findings

**Tipping by gender: no detectable difference.**

| Group | n | Mean tip |
| --- | --- | --- |
| Male | 157 | $3.09 |
| Female | 87 | $2.83 |

Welch's t-test gives t = 1.49, p = 0.138. At alpha = 0.05 the null hypothesis
holds, so the 26 cent gap in the sample is not distinguishable from noise.

Welch's rather than Student's because the groups are unbalanced (157 against 87)
and their variances differ, which is exactly the case where assuming equal
variance inflates the chance of a false positive.

**Bill total explains about 46% of tip variation.**

Tip = 0.105 x total_bill + 0.920, R² = 0.457.

So roughly 10 cents of tip per dollar of bill, with a 92 cent floor. The
remaining 54% of variation comes from something this model does not capture:
party size, service quality, day of week, or individual habit.

## Running it

```bash
pip install pandas seaborn matplotlib scipy scikit-learn
python tipping_analysis.py
```

Data is pulled from the seaborn repository at runtime, so nothing is stored locally.

## What this does not do

The R² is measured on the same data the model was fitted to, which is fine for a
descriptive fit on 244 rows but is not an estimate of predictive performance on
unseen bills. There is no train/test split and no cross-validation.

The t-test compares mean tip in absolute dollars. Since tip scales with bill
size, a fairer comparison would use tip percentage, and the groups may differ in
average bill size in a way this does not control for.

`size`, `day`, `time` and `smoker` are left out of the regression entirely. A
multivariate model would almost certainly explain more than 46%.

The dataset is a teaching set that ships with seaborn, not a sample I collected.