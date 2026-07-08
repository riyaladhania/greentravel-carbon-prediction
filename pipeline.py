import pandas as pd

# Load all the files
pub = pd.read_csv('public_trip_data.csv')
priv = pd.read_csv('private_trip_data.csv')
pub_log = pd.read_csv('public_trip_event_log.csv')
priv_log = pd.read_csv('private_trip_event_log.csv')
pub_attrs = pd.read_csv('public_trip_event_attributes.csv')
priv_attrs = pd.read_csv('private_trip_event_attributes.csv')
sample_sub = pd.read_csv('sample_submission.csv')

# Print shapes so we can confirm everything loaded correctly
print("public_trip_data:", pub.shape)
print("private_trip_data:", priv.shape)
print("public_trip_event_log:", pub_log.shape)
print("private_trip_event_log:", priv_log.shape)
print("public_trip_event_attributes:", pub_attrs.shape)
print("private_trip_event_attributes:", priv_attrs.shape)
print("sample_submission:", sample_sub.shape)

# Convert timestamp text into real datetime objects
def parse_ts(log):
    log = log.copy()
    log['EventTimestamp'] = pd.to_datetime(log['EventTimestamp'], format='%m/%d/%Y %H:%M:%S')
    return log

pub_log = parse_ts(pub_log)
priv_log = parse_ts(priv_log)

print("\nTimestamps converted. Sample:")
print(pub_log.head(3))
print("\nData type of EventTimestamp column:", pub_log['EventTimestamp'].dtype)

# Events that indicate something went "off-script" during the trip
CHANGE_EVENTS = ['Flight Change', 'Hotel Change', 'Mode of Transportation Change',
                 'Vehicle Change', 'Itinerary Edit', 'Flight Cancellation',
                 'Rental Cancellation', 'Train Cancellation', 'Missed Flight',
                 'Missed Train', 'Missed Pickup', 'Flight Delay', 'Train Delay',
                 'Travel Delay', 'Trip Extension', 'Ticket Reissued']
DELAY_EVENTS = ['Flight Delay', 'Train Delay', 'Travel Delay']

def build_event_features(log):
    # Pivot: each TripID becomes one row, each event type becomes a column
   
    piv = log.pivot_table(index='TripID', columns='EventName',
                           values='EventTimestamp', aggfunc='first')
    feats = pd.DataFrame(index=piv.index)

    # Booking lead time: days between booking transport and departing
    if 'Take Departure Flight' in piv.columns and 'Take Departure Train' in piv.columns:
        dep_time = piv['Take Departure Flight'].fillna(piv['Take Departure Train'])
    elif 'Take Departure Flight' in piv.columns:
        dep_time = piv['Take Departure Flight']
    else:
        dep_time = piv['Take Departure Train']

    if 'Book Mode of Transportation' in piv.columns:
        feats['booking_lead_days'] = (dep_time - piv['Book Mode of Transportation']).dt.total_seconds() / 86400

    if 'Submit Travel Request' in piv.columns and 'Travel Request Approved' in piv.columns:
        feats['approval_wait_hours'] = (piv['Travel Request Approved'] - piv['Submit Travel Request']).dt.total_seconds() / 3600

    feats['had_manager_preapproval'] = piv['Manager Preapproved'].notna().astype(int) if 'Manager Preapproved' in piv.columns else 0

    feats['n_change_events'] = log.groupby('TripID')['EventName'].apply(lambda x: x.isin(CHANGE_EVENTS).sum())
    feats['n_delay_events'] = log.groupby('TripID')['EventName'].apply(lambda x: x.isin(DELAY_EVENTS).sum())
    feats['total_steps'] = log.groupby('TripID')['StepOrder'].max()

    feats = feats.reset_index()
    for c in ['n_change_events', 'n_delay_events', 'had_manager_preapproval']:
        feats[c] = feats[c].fillna(0)
    return feats

pub_ev = build_event_features(pub_log)
priv_ev = build_event_features(priv_log)

print("\nEvent features built. Sample from public data:")
print(pub_ev.head())
print("\nShape:", pub_ev.shape)

def build_attr_features(attrs):
    feats = pd.DataFrame()
    feats['TripID'] = attrs['TripID']
    feats['had_expense_denial'] = attrs['ExpenseDenialReason'].notna().astype(int)
    feats['had_transport_cancellation'] = attrs['ReasonForTransportCancellation'].notna().astype(int)
    feats['had_new_hotel'] = attrs['NewHotelSelection'].notna().astype(int)
    feats['had_transport_change'] = attrs['ReasonForTransportationChange'].notna().astype(int)
    feats['had_delay_reason'] = attrs['ReasonForDelay'].notna().astype(int)
    feats['days_preapproved'] = pd.to_numeric(attrs['DaysPreapproved'], errors='coerce').fillna(0)
    feats['reimbursement_amount'] = pd.to_numeric(attrs['ExpenseReimbursementAmount'], errors='coerce').fillna(0)
    return feats

pub_attr_feats = build_attr_features(pub_attrs)
priv_attr_feats = build_attr_features(priv_attrs)

print("\nAttribute features built. Sample:")
print(pub_attr_feats.head())

# 'route' = Departure country + Arrival country. This turned out to be one of
# the strongest predictors — some routes are almost always high-carbon.
pub['route'] = pub['DepartureLocationCountry'] + '-' + pub['ArrivalLocationCountry']
priv['route'] = priv['DepartureLocationCountry'] + '-' + priv['ArrivalLocationCountry']

train = pub.merge(pub_ev, on='TripID', how='left').merge(pub_attr_feats, on='TripID', how='left')
test = priv.merge(priv_ev, on='TripID', how='left').merge(priv_attr_feats, on='TripID', how='left')

print("\nMerged train shape:", train.shape)
print("Merged test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())

from sklearn.preprocessing import LabelEncoder

# These columns are BANNED by competition rules - they leak the answer directly
BANNED = ['Departure_CO2e', 'Return_CO2e', 'Hotel_CO2e', 'Spend_CO2e', 'TotalCO2e', 'HighCarbon']
# These are just ID/location columns too specific to be useful patterns
ID_COLS = ['TripID', 'EmployeeNumber', 'DepartureLocationCity', 'ArrivalLocationCity']

feature_cols = [c for c in train.columns if c not in BANNED + ID_COLS]
cat_cols = ['DepartureLocationCountry', 'ArrivalLocationCountry', 'ShippingTypeDescription',
            'Purpose', 'OutOfPolicy', 'BusinessUnit', 'route']
num_cols = [c for c in feature_cols if c not in cat_cols]

X = train[feature_cols].copy()
y = train['HighCarbon'].copy()
X_test = test[feature_cols].copy()

# Convert text categories into numbers 
encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[c].astype(str), X_test[c].astype(str)], axis=0)
    le.fit(combined)
    X[c] = le.transform(X[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))
    
for c in num_cols:
    med = X[c].median()
    X[c] = X[c].fillna(med)
    X_test[c] = X_test[c].fillna(med)

print("\nFinal feature columns used:", feature_cols)
print("\nAny missing values left in X?", X.isnull().sum().sum())
print("Any missing values left in X_test?", X_test.isnull().sum().sum())
print("\nX shape:", X.shape, "| y shape:", y.shape, "| X_test shape:", X_test.shape)

from sklearn.model_selection import train_test_split
import lightgbm as lgb

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced',   # HighCarbon is ~25%/75% imbalanced, this compensates
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
)

print("\nModel training complete!")

from sklearn.metrics import roc_auc_score, f1_score, classification_report

val_probs = model.predict_proba(X_val)[:, 1]
val_preds = (val_probs >= 0.5).astype(int)

print("\n--- Validation Results ---")
print(f"ROC-AUC: {roc_auc_score(y_val, val_probs):.4f}")
print(f"F1 Score: {f1_score(y_val, val_preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_preds))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- Top 10 Most Important Features ---")
print(importances.head(10))

test_probs = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'TripID': test['TripID'],
    'HighCarbon': test_probs
})


submission = sample_sub[['TripID']].merge(submission, on='TripID', how='left')

submission.to_csv('submission.csv', index=False)

print("\n--- Submission saved! ---")
print("Shape:", submission.shape)
print("Sample submission shape:", sample_sub.shape)
print("TripIDs match sample_submission:", set(sample_sub['TripID']) == set(submission['TripID']))
print("\nFirst 10 rows:")
print(submission.head(10))