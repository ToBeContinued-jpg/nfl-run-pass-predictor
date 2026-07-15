
# NFL Run or Pass Predictor

An interactive machine-learning application that predicts whether an
NFL offense is more likely to call a run or pass based on the pre-snap
game situation.

## Live Application

The Streamlit application link will be added here after deployment.

## Project Question

To what extent can pre-snap game information predict whether an NFL
offense will call a run or pass?

## Model Inputs

- Down
- Yards to first down
- Quarter
- Time remaining in the quarter
- Down-and-distance interaction
- Score differential
- Field position
- Offensive timeouts remaining
- Defensive timeouts remaining

## Method

NFL play-by-play data from the 2021 through 2024 seasons was cleaned to
include standard run and pass plays. Quarterback kneels and spikes were
removed.

The model was trained on the 2021 through 2023 seasons and evaluated on
the held-out 2024 season. A logistic-regression pipeline performs
categorical encoding, numeric scaling, and classification.

## Results

| Model | Accuracy | Log Loss |
|---|---:|---:|
| Majority-class baseline | 57.0% | — |
| Initial logistic regression | 63.2% | 0.633 |
| Final feature model | 66.2% | 0.608 |

The score differential produced the largest improvement among the
additional features. Field position and timeout information produced
smaller improvements, particularly in probability quality.

## Limitations

The model does not know the offensive personnel, formation, coaching
strategy, weather, injuries, or the specific teams involved. Its
prediction represents historical tendencies rather than certainty
about an individual play.

## Tools

- Python
- pandas
- scikit-learn
- Streamlit
- NFL play-by-play data

## AI Assistance

AI tools were used for initial code scaffolding and debugging support.
I performed the data cleaning, feature engineering, model experiments,
evaluation, interpretation, and application integration.
