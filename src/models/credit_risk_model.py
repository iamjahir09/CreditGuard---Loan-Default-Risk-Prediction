from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,precision_score, recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from src.models.base_model import BaseModel

class credit_risk_model (BaseModel):
    def __init__(self,preprocessor,threshold=0.3):
        self.threshold = threshold
        self.pipeline = Pipeline([
            ('Preprocessor', preprocessor),
            ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
        ])

    def train(self,X,y):
        self.pipeline.fit(X,y)

    def predict_proba(self,X):
        return self.pipeline.predict_proba(X)[:, 1]
        
    def predict(self,X):
        proba = self.predict_proba(X)
        return (proba > self.threshold).astype(int)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
                'confusion_matrix': confusion_matrix(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred)
        }
