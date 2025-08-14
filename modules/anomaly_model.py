# modules/anomaly_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def run_anomaly_detection(df, model_name="Logistic Regression"):
    if df is None or df.empty:
        return pd.DataFrame({"오류": ["데이터가 비어있어 이상탐지 불가"]})

    df = df.dropna()
    df_encoded = df.copy()

    # 간단한 전처리: 범주형은 라벨 인코딩
    for col in df.columns:
        if df[col].dtype == 'object':
            df_encoded[col] = LabelEncoder().fit_transform(df[col])

    if df_encoded.shape[1] < 2:
        return pd.DataFrame({"오류": ["유효한 입력 특성이 부족합니다."]})

    X = df_encoded.iloc[:, :-1]
    y = df_encoded.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "Logistic Regression":
        return logistic_regression_anomaly(X_train, X_test, y_train)

    elif model_name == "XGBoost":
        return xgboost_anomaly(X_train, X_test, y_train)

    elif model_name == "AutoEncoder":
        return autoencoder_anomaly(X_train, X_test, y_train)

    else:
        return pd.DataFrame({"오류": ["알 수 없는 모델입니다."]})

def logistic_regression_anomaly(X_train, X_test, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result_df = pd.DataFrame(X_test)
    result_df["prediction"] = preds
    return result_df

def xgboost_anomaly(X_train, X_test, y_train):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return pd.DataFrame({"오류": ["xgboost가 설치되어 있지 않습니다."]})

    num_classes = len(np.unique(y_train))

    model = XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    result_df = pd.DataFrame(X_test)
    result_df["prediction"] = preds
    return result_df

def autoencoder_anomaly(X_train, X_test, y_train):
    try:
        from keras.models import Model
        from keras.layers import Input, Dense
    except ImportError:
        return pd.DataFrame({"오류": ["Keras가 설치되어 있지 않습니다."]})

    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_train, X_train, epochs=20, batch_size=16, verbose=0)

    X_test_pred = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

    result_df = pd.DataFrame(X_test)
    result_df["reconstruction_error"] = mse
    return result_df