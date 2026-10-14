import pandas as pd
from D1 import load_data, train_test_split_ts
from build_features import build_features, CATEGORICAL_FEATURES
from D5_lightgbm_backtest import to_categorical, make_model

df = load_data('.data/2024_smd_hourly.xlsx')
y = df['RT_Demand'].dropna()

X_all = build_features(df.loc[y.index], feature_set='time_lag_rolling')
split_date = pd.Timestamp('2024-07-01')
X_train, X_test, y_train, y_test = train_test_split_ts(X_all, y, split_date)

# 对齐 dropna
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
y_test = y_test.loc[X_test.index]

X_train, X_test = to_categorical(X_train, X_test, CATEGORICAL_FEATURES)

model = make_model()
model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(imp)