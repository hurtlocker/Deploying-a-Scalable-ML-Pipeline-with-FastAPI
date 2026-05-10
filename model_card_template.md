# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a gradient boosting classifier trained to predict if an individuals annual income exceeds $50K based on US census demographic data. It uses Python 3.12, scikit-learn 1.5.1, and GradientBoostingClassifier with the following parameters: `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`, `random_state=42`.

## Intended Use
Intended for educational use only to demonstrate ML pipeline deployment via FastAPI.


## Training Data
Training data was the UCI Adult Census Income Dataset containing over 32K records from the 1994 Census. 

## Evaluation Data
Evaluated on a 20% test set of approx 6.5k records. Slice-level performance was computed across unique values with full results available in `slice_output.txt`.


## Metrics

Performance is reported using precision, recall, and F1 score (fbeta, beta=1).

Overall test set performance:
- **Precision:** 0.8030
- **Recall:** 0.6225
- **F1 Score:** 0.7013

Notable slice-level findings: females underperform males (F1: 0.6194 vs. 0.7143),
and Black and Amer-Indian-Eskimo subgroups underperform White and Asian-Pac-Islander
subgroups. Higher education levels generally correlate with higher F1 scores.

## Ethical Considerations
The dataset contains sensitive attributes including race, sex, and native country.
Slice-level metrics reveal performance disparities across demographic groups. This
model should not inform decisions in employment, credit, or housing contexts.


## Caveats and Recommendations
Training data is from 1994 and may not reflect current conditions. Slices with small
sample sizes should be interpreted with caution. 