from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

@dataclass
class Model:
    model: XGBClassifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
