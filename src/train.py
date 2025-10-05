import os
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor


def train_models(X_train_c, X_test_c, y_train_c, y_test_c,
                 X_train_r, X_test_r, y_train_r, y_test_r,
                 scaler, features, model_dir="models"):


    os.makedirs(model_dir, exist_ok=True)

    # ---------------- Classification ----------------
    print("\n🚀 Training Classification Model (XGBoost + GridSearch)...")
    try:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.05, 0.1]
        }

        grid_clf = GridSearchCV(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            ),
            param_grid, cv=3, scoring="accuracy",
            verbose=1, n_jobs=-1
        )

        grid_clf.fit(X_train_c, y_train_c)
        best_clf = grid_clf.best_estimator_

        y_pred_c = best_clf.predict(X_test_c)
        acc = accuracy_score(y_test_c, y_pred_c)
        f1 = f1_score(y_test_c, y_pred_c, average="weighted")

        print("✅ Best Classifier Params:", grid_clf.best_params_)
        print(f"🎯 Accuracy: {acc:.4f}")
        print(f"🔥 F1-score: {f1:.4f}")
        print("\n📑 Classification Report:\n", classification_report(y_test_c, y_pred_c))

    except Exception as e:
        print("❌ Error in Classification Training:", str(e))
        best_clf = None

    # ---------------- Regression ----------------
    print("\n🚀 Training Regression Model (XGBoost)...")
    try:
        reg = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, objective="reg:squarederror"
        )
        reg.fit(X_train_r, y_train_r)
        y_pred_r = reg.predict(X_test_r)

        mse = mean_squared_error(y_test_r, y_pred_r)
        r2 = r2_score(y_test_r, y_pred_r)

        print(f"📉 MSE: {mse:.4f}")
        print(f"📊 R² Score: {r2:.4f}")

    except Exception as e:
        print("❌ Error in Regression Training:", str(e))
        reg = None

    # ---------------- Save Models ----------------
    print("\n💾 Saving Models...")
    if best_clf:
        joblib.dump(best_clf, f"{model_dir}/classifier.pkl")
    if reg:
        joblib.dump(reg, f"{model_dir}/regressor.pkl")

    joblib.dump(scaler, f"{model_dir}/scaler.pkl")
    joblib.dump(features, f"{model_dir}/features.pkl")

    print("✅ Models saved successfully in:", model_dir)

    return best_clf, reg
