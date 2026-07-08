# greentravel-carbon-prediction
ML classifier predicting high-carbon business trips - Summer Analytics 2026

# GreenTravel Intelligence Challenge - High-Carbon Trip Prediction

Predicts whether a business trip will be classified as "HighCarbon" using LightGBM.
Built for Summer Analytics 2026 (Consulting & Analytics Club, IIT Guwahati).

## Approach
- EDA showed transport mode and route (departure-arrival country) are the
  strongest predictors of carbon emissions
- Engineered features from event logs: booking lead time, approval wait time,
  number of change/delay events during the trip
- Engineered features from event attributes: expense denial flags, transport
  cancellation flags, reimbursement amounts
- Excluded all banned target-leaking columns (Departure_CO2e, Return_CO2e,
  Hotel_CO2e, Spend_CO2e, TotalCO2e)

## Model
LightGBM Classifier with class_weight='balanced' (HighCarbon is ~25% of data)

## Results
- Validation ROC-AUC: 0.9994
- Validation F1 Score: 0.987

The high score reflects that carbon emissions are calculated largely from
distance (route) and transport mode, both of which are available features -
making the problem close to fully separable using only allowed inputs.
Feature importance confirms this: ArrivalLocationCountry, route, and
ShippingType dominate, while business context features (BusinessUnit, Purpose)
contribute very little.

## Files
- `pipeline.py` - full pipeline: data loading, feature engineering, model
  training, evaluation, and submission generation
- `submission.csv` - final predictions on the private test set
