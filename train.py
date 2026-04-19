import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,f1_score, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
import joblib

#Analysing data
df=pd.read_csv('data\HR-employee-Attrition.csv')
print(df.shape)
print(df.sample(9))
df=df.drop(['EmployeeCount'],axis=1)
print('Employee Count dropped')
print(df['Attrition'].value_counts())
df['Attrition']=df['Attrition'].map({'Yes':1,'No':0})
df['target']=df['Attrition']
df.drop(['Attrition'],axis=1)
print(df.info())

# Visualizing Attrition 
sns.countplot(x='target',data=df)
plt.title('Attrition count')
plt.show()

print(df['Gender'].unique())
df=df.drop(['Over18'],axis=1)
print("Over18 dropped")
df['OverTime']=df['OverTime'].map({'Yes':1,'No':0})
df['Gender']=df['Gender'].replace({'Male':1,'Female':0})
print('Overtime and gender changed to 0 and 1')

#Countplot attrition according to gender
df=df.drop(['EmployeeNumber'],axis=1)
df=df.drop(['Attrition'],axis=1)

#represents the gap between salaries therefore need standardization
sns.boxplot(x='target',y='MonthlyIncome',data=df)
plt.title("Monthly income based on attrition")
plt.show()

#Distplot to understand monthly income of different employees
sns.displot(x='MonthlyIncome',data=df,kde=True)
plt.title("Monthly income")
plt.show()

#Visualization for overtime
sns.countplot(x='OverTime',hue='target',data=df)
plt.title("Attrition based on overtime")
plt.show()

df=df.drop(['StandardHours'],axis=1)
print('Satndard hours dropped')

#Heatmap to understand correlation between data
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

X=df.drop(['target'],axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=42, test_size=0.2)
categorical_cols=X.select_dtypes(include=['object']).columns.tolist()
numerical_cols=X.select_dtypes(include=['int64']).columns.tolist()

preprocessor=ColumnTransformer([
    ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_cols),
    ('num',StandardScaler(),numerical_cols)
]
)
model={
    'Logistic Regression':make_pipeline(
        preprocessor,
        LogisticRegression(max_iter=500)
    ),
    'Random Forest':make_pipeline(
        preprocessor,
        RandomForestClassifier(n_estimators=500)
    ),
    'XGBoost':make_pipeline(
        preprocessor,
        XGBClassifier()
    ),
    'SVM':make_pipeline(
        preprocessor,
        SVC(kernel='rbf',probability=True)
    )
}

trained_models={}
result={}
for name,pipeline in model.items():
  print(f'\n Training {name} model')
  pipeline.fit(X_train,y_train)
  y_pred=pipeline.predict(X_test)
  y_proba=pipeline.predict_proba(X_test)[:,1]

  accuracy =accuracy_score(y_test,y_pred)
  precision =precision_score(y_test,y_pred)
  recall =recall_score(y_test,y_pred)
  f1 =f1_score(y_test,y_pred)
  roc_auc =roc_auc_score(y_test,y_proba)
  result[name]={
      'accuracy':accuracy,
      'precision':precision,
      'recall':recall,
      'f1':f1,
      'roc_auc':roc_auc
  }
  trained_models[name]= pipeline
  print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
  print(f"F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

#Checking for important features-----------------------------------------------------------------------------------------------
cat_feature_names=preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
num_feature_names=numerical_cols
all_feature_names=list(cat_feature_names)+num_feature_names
print("\n All feature names\n",all_feature_names)

#Using random forest for feature importance
rf_pipeline=model['Random Forest']
rf_pipeline.fit(X_train,y_train)
importance_rf=pd.Series(
    rf_pipeline.named_steps['randomforestclassifier'].feature_importances_,
    index=all_feature_names
)
print("\n Importantance of features in Random forest\n",importance_rf.sort_values(ascending=False))

importance_rf.sort_values(ascending=False).plot(kind='bar',figsize=(15,5))
plt.title("Feature Importance in Random Forest")
plt.show()

#Using XGBoost to find feature importance
xgboost_pipeline=model['XGBoost']
xgboost_pipeline.fit(X_train,y_train)
importance_xgb=pd.Series(
    xgboost_pipeline.named_steps['xgbclassifier'].feature_importances_,
    index=all_feature_names
)
print("\n Importantance of features in XGB\n",importance_xgb.sort_values(ascending=False))

importance_xgb.sort_values(ascending=False).plot(kind='bar',figsize=(15,5))
plt.title("Feature Importance in xgboost")
plt.show()

important_transformed=importance_xgb[importance_xgb >0.01].index.tolist()
print("Important transformed features",important_transformed)

#Mapping back to original columns
important_original = set()

for feature in important_transformed:
    for col in categorical_cols:
        if feature.startswith(col):
            important_original.add(col)
    for col in numerical_cols:
        if feature == col:
            important_original.add(col)

important_original = list(important_original)
print("Original columns to keep:", important_original)
print("Count:", len(important_original))

dropped_cols=[c for c in categorical_cols +numerical_cols if c not in important_original]
print("count Dropped columns:", len(dropped_cols))
print("Dropped columns:", dropped_cols)

new_models={}
new_categorical_cols=[c for c in categorical_cols if c in important_original]
new_numerical_cols=[c for c in numerical_cols if c in important_original]
new_preprocessor=ColumnTransformer([
    ('cat',OneHotEncoder(handle_unknown='ignore'),new_categorical_cols),
    ('num',StandardScaler(),new_numerical_cols)
])
new_models={
    'Logistic Regression':make_pipeline(
        new_preprocessor,
        LogisticRegression(max_iter=500)
    ),
    'Random Forest':make_pipeline(
        new_preprocessor,
        RandomForestClassifier(n_estimators=500)
    ),
    'XGBoost':make_pipeline(
        new_preprocessor,
        XGBClassifier()
    ),
    'SVM':make_pipeline(
        new_preprocessor,
        SVC(kernel='rbf',probability=True)
    )}

new_trained_models = {}
new_result = {}

X_train_new = X_train[important_original]
X_test_new = X_test[important_original]

for new_name, new_pipeline in new_models.items():
    print(f'\nTraining {new_name} model')
    new_pipeline.fit(X_train_new, y_train)
    new_y_pred = new_pipeline.predict(X_test_new)
    new_y_proba = new_pipeline.predict_proba(X_test_new)[:, 1]

    new_accuracy = accuracy_score(y_test, new_y_pred)
    new_precision = precision_score(y_test, new_y_pred)
    new_recall = recall_score(y_test, new_y_pred)
    new_f1 = f1_score(y_test, new_y_pred)
    new_roc_auc = roc_auc_score(y_test, new_y_proba)

    new_result[new_name] = {
        'accuracy': new_accuracy,
        'precision': new_precision,
        'recall': new_recall,
        'f1': new_f1,
        'roc_auc': new_roc_auc
    }
    new_trained_models[new_name] = new_pipeline
    print(f"Accuracy: {new_accuracy:.4f} | Precision: {new_precision:.4f} | Recall: {new_recall:.4f}")
    print(f"F1-Score: {new_f1:.4f} | ROC-AUC: {new_roc_auc:.4f}")

    print(y_train.value_counts())
print(y_train.value_counts(normalize=True) * 100)
print(y_train.value_counts())
print(df['target'].value_counts())

#Now to balance data
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
smote_models={
    'Logistic Regression':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        LogisticRegression(max_iter=500)
    ),
    'Random Forest':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        RandomForestClassifier(n_estimators=500)
    ),
    'XGBoost':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        XGBClassifier()
    ),
    'SVM':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        SVC(kernel='rbf',probability=True)
    )
}
#Applying SMOTE to balance Imbalance data
smote_results={}
smote_trained_models={}
for name_smote,pipeline_smote in smote_models.items():
  print(f"\nTraining {name_smote} model")
  pipeline_smote.fit(X_train_new,y_train)
  smote_y_pred=pipeline_smote.predict(X_test_new)
  smote_y_proba=pipeline_smote.predict_proba(X_test_new)[:,1]

  smote_accuracy=accuracy_score(y_test,smote_y_pred)
  smote_precision=precision_score(y_test,smote_y_pred)
  smote_recall=recall_score(y_test,smote_y_pred)
  smote_f1=f1_score(y_test,smote_y_pred)
  smote_roc_auc=roc_auc_score(y_test,smote_y_proba)

  smote_results[name_smote]={
      'accuracy':smote_accuracy,
      'precision':smote_precision,
      'recall':smote_recall,
      'f1':smote_f1,
      'roc_auc':smote_roc_auc
  }
  smote_trained_models[name_smote]=pipeline_smote
  print(f'Accuracy:{smote_accuracy:.4f} | Precision:{smote_precision:.4f}| Recall:{smote_recall:.4f}')
  print(f"F1:{smote_f1:.4f}| ROC-AUC:{smote_roc_auc:.4f}")

#Balanced data using SMOTE and class weight
balanced_models={
    'Logistic Regression':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        LogisticRegression(max_iter=500, class_weight='balanced')
    ),
    'Random Forest':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        RandomForestClassifier(n_estimators=500, class_weight='balanced')
    ),
    'XGBoost':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        XGBClassifier(scale_pos_weight=5)
    ),
    'SVM':imb_make_pipeline(
        new_preprocessor,
        SMOTE(random_state=42),
        SVC(kernel='rbf',probability=True, class_weight='balanced')
    )
}
balanced_results={}
balanced_trained_models={}
for name_balanced,pipeline_balanced in balanced_models.items():
  print(f"\nTraining {name_balanced} model")
  pipeline_balanced.fit(X_train_new,y_train)
  balanced_y_pred=pipeline_balanced.predict(X_test_new)
  balanced_y_proba=pipeline_balanced.predict_proba(X_test_new)[:,1]

  balanced_accuracy=accuracy_score(y_test,balanced_y_pred)
  balanced_precision=precision_score(y_test,balanced_y_pred)
  balanced_recall=recall_score(y_test,balanced_y_pred)
  balanced_f1=f1_score(y_test,balanced_y_pred)
  balanced_roc_auc=roc_auc_score(y_test,balanced_y_proba)

  balanced_results[name_balanced]={
      'accuracy':balanced_accuracy,
      'precision':balanced_precision,
      'recall':balanced_recall,
      'f1':balanced_f1,
      'roc_auc':balanced_roc_auc
  }
  balanced_trained_models[name_balanced]=pipeline_balanced
  print(f'Accuracy:{balanced_accuracy:.4f} | Precision:{balanced_precision:.4f}| Recall:{balanced_recall:.4f}')
  print(f"F1:{balanced_f1:.4f}| ROC-AUC:{balanced_roc_auc:.4f}")

# Parameter grids
#Applying RandomsearchCV to find best parameters
lr_params = {
    'logisticregression__C': uniform(0.01, 10),
    'logisticregression__solver': ['lbfgs', 'saga'],
    'logisticregression__penalty': ['l2']
}

rf_params = {
    'randomforestclassifier__n_estimators': randint(100, 500),
    'randomforestclassifier__max_depth': [None, 5, 10, 15, 20],
    'randomforestclassifier__min_samples_split': randint(2, 10),
    'randomforestclassifier__min_samples_leaf': randint(1, 5)
}

xgb_params = {
    'xgbclassifier__n_estimators': randint(100, 500),
    'xgbclassifier__max_depth': randint(3, 10),
    'xgbclassifier__learning_rate': uniform(0.01, 0.3),
    'xgbclassifier__subsample': uniform(0.6, 0.4),
    'xgbclassifier__colsample_bytree': uniform(0.6, 0.4)
}

svm_params = {
    'svc__C': uniform(0.1, 10),
    'svc__gamma': ['scale', 'auto'],
    'svc__kernel': ['rbf', 'sigmoid']
}

# Pipelines
lr_pipeline = imb_make_pipeline(
    new_preprocessor,
    SMOTE(random_state=42),
    LogisticRegression(max_iter=500, class_weight='balanced')
)

rf_pipeline = imb_make_pipeline(
    new_preprocessor,
    SMOTE(random_state=42),
    RandomForestClassifier(n_estimators=500, class_weight='balanced')
)

xgb_pipeline = imb_make_pipeline(
    new_preprocessor,
    SMOTE(random_state=42),
    XGBClassifier(scale_pos_weight=5)
)

svm_pipeline = imb_make_pipeline(
    new_preprocessor,
    SMOTE(random_state=42),
    SVC(kernel='rbf', probability=True)
)

# Randomized Search
tuned_models = {}
tuned_results = {}

searches = {
    'Logistic Regression': (lr_pipeline, lr_params),
    'Random Forest': (rf_pipeline, rf_params),
    'XGBoost': (xgb_pipeline, xgb_params),
    'SVM': (svm_pipeline, svm_params)
}

for name, (pipeline, params) in searches.items():
    print(f'\nTuning {name}...')
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=20,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train_new, y_train)

    best_model = search.best_estimator_
    y_pred_tuned = best_model.predict(X_test_new)
    y_proba_tuned = best_model.predict_proba(X_test_new)[:, 1]

    tuned_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'f1': f1_score(y_test, y_pred_tuned),
        'roc_auc': roc_auc_score(y_test, y_proba_tuned)
    }
    tuned_models[name] = best_model

    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy: {tuned_results[name]['accuracy']:.4f} | Precision: {tuned_results[name]['precision']:.4f} | Recall: {tuned_results[name]['recall']:.4f}")
    print(f"F1: {tuned_results[name]['f1']:.4f} | ROC-AUC: {tuned_results[name]['roc_auc']:.4f}")

# Final best model
best_model = tuned_models['XGBoost']

print("Best Model: XGBoost (Tuned with SMOTE)")
print("-" * 70)

# Compare all blocks
print("\nBlock 1 - Original 32 Features:")
print(pd.DataFrame(result).T.round(4))

print("\nBlock 2 - Feature Selected 29 Features:")
print(pd.DataFrame(new_result).T.round(4))

print("\nBlock 3 - SMOTE Applied:")
print(pd.DataFrame(balanced_results).T.round(4))

print("\nBlock 4 - Tuned Models (Final):")
print(pd.DataFrame(tuned_results).T.round(4))

#Saving models 
joblib.dump(best_model, 'models/xgboost_attrition_model.joblib')
joblib.dump(important_original, 'models/selected_features.joblib')
print("Model saved successfully!")

#Plotting confusion matrix 
y_pred_final = best_model.predict(X_test_new)
y_proba_final = best_model.predict_proba(X_test_new)[:, 1]
print("Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['No Attrition', 'Attrition']))
cm=confusion_matrix(y_test,y_pred_final)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues',
            xticklabels=['No Attrition','Attrition'],
            yticklabels=['No Attrition','Attrition'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion matrix XGBoost ')
plt.savefig('models/confusion_matrix.png', dpi=100)
plt.tight_layout()

#Plotting ROC-AUC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_final)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost (Tuned)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=100)
plt.show()