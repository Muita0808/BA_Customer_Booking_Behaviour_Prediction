# Predicting Customer Buying Behaviour — British Airways

## Background

Customers today have more information and more choice than ever, which has changed the airline
booking cycle. Waiting until a customer is at the airport to try to win their business is too
late — airlines need to be proactive and identify likely buyers before they travel. This project
uses historical booking data and a predictive model to support that shift from reactive to
proactive customer acquisition.

The quality of the underlying data matters as much as the model itself, so a meaningful part of
this project is spent preparing the data properly — and once a model is trained, interpreting
*how* predictive it actually is, not just reporting a single accuracy number.

## Task

This project follows a three-part brief:

1. **Explore and prepare the dataset** — understand the raw booking data, engineer new features
   where useful, and get the data into a shape suitable for modeling.
2. **Train a machine learning model** — predict `booking_complete` (whether a customer completes
   a booking) using an algorithm that also exposes feature-level importance, so the results can
   be interpreted, not just used as a black box.
3. **Evaluate the model and present findings** — validate performance with cross-validation and
   metrics appropriate to the data, visualize what drove the model's predictions, and summarize
   the findings in a single slide for a non-technical audience (a manager).

## Dataset

50,000 customer booking records (`customer_booking.csv`), each with pre-purchase and flight
details — how far in advance the booking was made, trip length, flight time, route, requested
add-ons, and more — plus the target, `booking_complete`.

## Approach

- **Cleaning:** removed 719 duplicate rows before splitting the data, to avoid the same
  observation leaking into both the training and test sets.
- **Feature engineering:** frequency-encoded the high-cardinality `route` (split into origin/
  destination) and `booking_origin` fields instead of one-hot encoding hundreds of rare
  categories; one-hot encoded the remaining low-cardinality categoricals.
- **Modeling:** Random Forest classifier, chosen specifically because it exposes feature
  importances for interpretation, as the brief calls for.
- **Handling class imbalance:** only ~15% of bookings in the dataset are completed. A default
  model reached 85% accuracy while catching almost none of the completed bookings (recall 0.09)
  — accuracy alone was hiding a model that wasn't actually useful. Tuning the decision threshold
  (rather than relying on the default 0.5 cutoff) raised recall to 0.60 and F1 to 0.41.
- **Validation:** confirmed the tuned result was stable, not a lucky split, using 5-fold
  cross-validation evaluated at the same tuned threshold (F1 consistent at 0.41–0.42 across
  folds).
- **Interpretation:** extracted and visualized Random Forest feature importances to identify
  what actually drives booking completion.

## Results

| Model / setting | Accuracy | Recall (completed) | F1 (completed) |
|---|---|---|---|
| Baseline (default threshold) | 0.850 | 0.09 | 0.15 |
| Class-balanced, tuned threshold (0.20) | 0.742 | 0.60 | 0.41 |
| Tuned threshold, 5-fold CV | — | — | 0.42 (± 0.01) |

**What drives completion:** booking timing and origin patterns — how far ahead a customer books,
trip length, flight hour, and how common their route/origin is — far more than add-on
preferences (extra baggage, seat choice, meals) or trip type.

## Findings summary

Accuracy is the wrong headline metric on this imbalanced target — it rewards a model for
mirroring the majority class rather than identifying likely buyers. Tuning the decision threshold
trades some accuracy for a large, validated gain in the model's ability to actually catch
completed bookings, which is the outcome that matters for a proactive acquisition strategy.

## Repo contents

- `BA_Booking_Prediction.ipynb` — full analysis (exploration, feature engineering, modeling,
  threshold tuning, cross-validation, feature importance)
- `feature_importance.png` — feature importance chart
- `BA_Booking_Prediction_Summary.pptx` — one-slide summary for a non-technical audience

## Tools

Python, pandas, scikit-learn, matplotlib.

---
*Completed as part of the British Airways Data Science virtual internship on Forage. The dataset
and task brief are provided by Forage/British Airways; the analysis and write-up are my own.*
