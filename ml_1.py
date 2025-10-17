# Author: Berkay Ekren
# Date: 2025-10-15
# Description: This script performs multi-omics data integration using the Early Integration strategy.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import necessary ml libraries
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay

# Get working directory
cwd = os.getcwd()
prd = "Projects/BIOAQUA_COST_CA22160-20251021"

# Get the medatada
metadata_df = pd.read_csv("{}/{}/data/metadata.csv".format(cwd, prd), sep="\t")

# Import data [1]
microbiome_df = pd.read_csv("{}/{}/data/microbiome.csv".format(cwd, prd), sep="\t")
metabolome_df = pd.read_csv("{}/{}/data/metabolome.csv".format(cwd, prd), sep="\t")

# Uncomment the below 2 lines to see the first few rows of the dataframes to see the file structure
#print(microbiome_df.head())
#print(metabolome_df.head())

# Check the distribution of the data with histograms
plt.figure(figsize=(14, 5))
sns.histplot(microbiome_df.iloc[:, 1:].values.flatten(), bins=50, color='blue', label='Microbiome', kde=True)
sns.histplot(metabolome_df.iloc[:, 1:].values.flatten(), bins=50, color='orange', label='Metabolome', kde=True)
plt.title('Distribution of Relative Abundance Values', fontsize=16)
plt.xlabel('Abundance')
plt.ylabel('Frequency')
plt.legend()
# Create the prd directory if it does not exist
prd_path = os.path.join(cwd, prd)
if not os.path.exists(prd_path):
    os.makedirs(prd_path)
plt.savefig(os.path.join(prd_path, "data_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()


print("--- Strategy 1: Early Integration ---")

# Set the first column as the index for each dataframe
microbiome_features = microbiome_df.set_index(microbiome_df.columns[0])
metabolome_features = metabolome_df.set_index(metabolome_df.columns[0])

# Transpose the dataframes so that rows are samples and columns are features
X_microbiome = microbiome_features.T
X_metabolome = metabolome_features.T

# Concatenate the dataframes horizontally (axis=1) to create a single feature matrix.
# This aligns the data by sample ID (the index).
early_integration_df = pd.concat([X_microbiome, X_metabolome], axis=1)

print("Shape of Microbiome data (samples, features):", X_microbiome.shape)
print("Shape of Metabolome data (samples, features):", X_metabolome.shape)
print("Shape of combined data for Early Integration:", early_integration_df.shape)

print("\n--- Early Integration DataFrame Head ---")
print(early_integration_df.head(2))

# Create target variables (y) from metadata. Set the first column (sample IDs) as the index of the metadata
metadata_indexed = metadata_df.set_index(metadata_df.columns[0])

# Align metadata with the feature dataframe to ensure correct sample order
aligned_metadata = metadata_indexed.reindex(early_integration_df.index)

# Create the classification target from the 'sampling_site' column
y_classification, class_labels = pd.factorize(aligned_metadata[aligned_metadata.columns[1]])
print(f"\nClassification target created from column: '{aligned_metadata.columns[1 ]}'")
print(f"Classes found: {class_labels.tolist()}")

# Split the data into training and testing sets
print("\nData splitting for testing and training...")


# For regression, we'll use the factorized labels as a placeholder. In practice, replace this with a real continuous variable.
y_regression = y_classification

# Split data for classification task
X_train_early_c, X_test_early_c, y_train_c, y_test_c = train_test_split(
    early_integration_df, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)
# Split data for regression task
X_train_early_r, X_test_early_r, y_train_r, y_test_r = train_test_split(
    early_integration_df, y_regression, test_size=0.2, random_state=42
)

# Display the shapes of the resulting datasets
print("\nShapes of the datasets after splitting:")
print("Classification Task:")
print("X_train_early_c:", X_train_early_c.shape, "X_test_early_c:", X_test_early_c.shape)
print("y_train_c:", y_train_c.shape, "y_test_c:", y_test_c.shape)
print("\nRegression Task:")
print("X_train_early_r:", X_train_early_r.shape, "X_test_early_r:", X_test_early_r.shape)
print("y_train_r:", y_train_r.shape, "y_test_r:", y_test_r.shape)

print("\nData successfully split for training and testing.")

# Feature Selection using Boruta
print ("\n--- Feature Selection using Boruta ---")

# Boruta needs a base estimator that provides feature importances. Random Forest is perfect.
rf_for_boruta = RandomForestClassifier(n_jobs=-1,class_weight='balanced',max_depth=5,random_state=42)

# n_estimators='auto' will let Boruta decide the number of trees and set verbose to 2 to see the progress
boruta_selector = BorutaPy(estimator=rf_for_boruta,n_estimators='auto',verbose=2, random_state=42)

# Boruta expects numpy arrays, so we use .values
# Note: Boruta can be slow on datasets with many features.
boruta_selector.fit(X_train_early_c.values, y_train_c)

# With small datasets, it's often useful to include tentative features.
selected_features_mask = boruta_selector.support_ | boruta_selector.support_weak_

X_train_selected_c = X_train_early_c.loc[:, selected_features_mask]
X_test_selected_c = X_test_early_c.loc[:, selected_features_mask]

# For the regression task, we'll use the same selected features for consistency
X_train_selected_r = X_train_early_r.loc[:, selected_features_mask]
X_test_selected_r = X_test_early_r.loc[:, selected_features_mask]

print(f"\nOriginal number of features: {X_train_early_c.shape[1]}")
print(f"Number of features selected by Boruta (Confirmed + Tentative): {X_train_selected_c.shape[1]}")

# Models for Classification (Early Integration)
# --- Hyperparameter Tuning with Cross-Validation for Multiple Models ---
print("\n--- Finding Best Model and Hyperparameters with GridSearchCV ---")

# Define the models and their parameter grids
# Calculate scale_pos_weight for imbalanced datasets
scale_pos_weight_value = (len(y_train_c) - sum(y_train_c)) / sum(y_train_c)
y_train_c = y_train_c.astype(int)
y_test_c = y_test_c.astype(int)

# Models for the main loop (XGBoost is now excluded)
models = {
    'LogisticRegression': LogisticRegression(random_state=42,
                                             max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'SVC': SVC(random_state=42,
               probability=True),
    'LightGBM': lgb.LGBMClassifier(random_state=42,
                                   verbose=-1), # verbose=-1 silences LGBM warnings
    'XGBoost': xgb.XGBClassifier(base_score=np.sum(y_train_c == 0)/len(y_train_c),
                                 objective='binary:logistic',
                                 booster='gbtree',
                                 tree_method='hist',
                                 random_state=42,
                                 scale_pos_weight=scale_pos_weight_value)
}

param_grids = {
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear'],
        'class_weight': ['balanced']
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 75, 100, 150],
        'max_depth': [2, 3, 5, 10],
        'min_samples_leaf': [1, 3],
        'class_weight': ['balanced']
    },
    'SVC': {
        'C': [0.01, 0.1, 1, 10, 50, 100],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced']
    },
    'LightGBM': {
        'n_estimators': [50, 75, 100, 150],
        'max_depth': [2, 3, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'class_weight': ['balanced'],
        'num_leaves': [7, 15, 31],
        'min_child_samples': [1, 2, 5, 10]
    },
    'XGBoost': {
        'n_estimators': [50, 75, 100, 150],
        'learning_rate': [0.001, 0.01, 0.1, 0.5],
        'gamma': [0, 0.1, 0.5],
        'max_depth': [3, 5],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0]
    }
}

# Set up the cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

best_score = -1
best_model_name = ""
best_estimator = None

# Loop through the models
for name, model in models.items():
    print(f"\n--- Tuning {name} ---")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train_selected_c, y_train_c)
    
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validated AUC score: {grid_search.best_score_:.4f}")
        
    best_model = grid_search.best_estimator_
    y_pred_final = best_model.predict(X_test_selected_c)
    y_proba_final = best_model.predict_proba(X_test_selected_c)[:, 1]
    
    # Plot Confusion Matrix, AUROC, and Feature Importance Side-by-Side
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4))

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test_c, y_pred_final)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix for {name}')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Plot 2: ROC Curve
    RocCurveDisplay.from_predictions(y_test_c, y_proba_final, ax=axes[1])
    axes[1].set_title(f'ROC Curve for {name}')
    
    # Plot 3: Feature Importance
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = best_model.coef_[0]

    if importances is not None:
        feature_names = X_train_selected_c.columns
        forest_importances = pd.Series(importances, index=feature_names)
        top_importances = forest_importances.abs().nlargest(15)
        
        top_importances.sort_values(ascending=True).plot.barh(ax=axes[2])
        axes[2].set_title(f'Top 15 Feature Importances for {name}')
        axes[2].set_xlabel('Importance')
    else:
        axes[2].text(0.5, 0.5, 'Feature importances not available', ha='center', va='center')
        axes[2].set_title(f'Feature Importances for {name}')
        axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(prd_path, f"early_integration_{name}_plots.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if grid_search.best_score_ > best_score:
        best_score = grid_search.best_score_
        best_model_name = name
        best_estimator = grid_search.best_estimator_

# Evaluate the Overall Best Model Found
if best_estimator is not None:
    print(f"\n--- Champion Model: {best_model_name} ---")
    print(f"Best CV AUC: {best_score:.4f}")

    # Evaluate the final, champion model on the held-out test set
    y_pred_final = best_estimator.predict(X_test_selected_c)
    y_proba_final = best_estimator.predict_proba(X_test_selected_c)[:, 1]

    final_accuracy = accuracy_score(y_test_c, y_pred_final)
    final_auc = roc_auc_score(y_test_c, y_proba_final)

    print("\nPerformance of Champion Model on Test Set:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  AUC: {final_auc:.4f}")

    print("\nClassification Report on Test Set:")
    print(classification_report(y_test_c, y_pred_final))

else:
    print("\nCould not determine a best model due to errors during tuning.")