# Author: Berkay Ekren
# Date: 2025-10-16
# Description: This script performs multi-omics data integration using the Late Integration strategy.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import necessary ml libraries
from sklearn.feature_selection import SelectFromModel # Uncomment if using SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import RocCurveDisplay

# Get working directory
cwd = os.getcwd()
prd = "Projects/BIOAQUA_COST_CA22160-20251021" # Change as needed

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
if not os.path.exists(prd):
    os.makedirs(prd)
plt.savefig(f"{prd}/data_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n--- Strategy 2: Late Integration Data Splitting ---")

# Create training and test dataframes for microbiome and metabolome raw data
microbiome_features = microbiome_df.set_index(microbiome_df.columns[0])
metabolome_features = metabolome_df.set_index(metabolome_df.columns[0])

# Transpose the dataframes so that rows are samples and columns are features
X_microbiome = microbiome_features.T
X_metabolome = metabolome_features.T

# Create target variables (y) from metadata
metadata_indexed = metadata_df.set_index(metadata_df.columns[0])

# Align metadata with the feature dataframe to ensure correct sample order
aligned_metadata = metadata_indexed.reindex(X_microbiome.index)

# Create the classification target from the 'sampling_site' column
y_classification, class_labels = pd.factorize(aligned_metadata[aligned_metadata.columns[1]])
print(f"\nClassification target created from column: '{aligned_metadata.columns[1 ]}'")
print(f"Classes found: {class_labels.tolist()}")

# Split microbiome data
X_micro_train, X_micro_test, y_train_late, y_test_late = train_test_split(
    X_microbiome, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

# Split metabolome data using the same indices from the first split
# We can achieve this by splitting the metabolome data with the same parameters.
X_metab_train, X_metab_test, _, _ = train_test_split(
    X_metabolome, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

# Display the shapes of the resulting datasets
print("\n--- Shapes of the datasets for Late Integration ---")
print("Microbiome Data:")
print("X_micro_train:", X_micro_train.shape, "| X_micro_test:", X_micro_test.shape)
print("\nMetabolome Data:")
print("X_metab_train:", X_metab_train.shape, "| X_metab_test:", X_metab_test.shape)
print("\nTarget Variable:")
print("y_train_late:", y_train_late.shape, "| y_test_late:", y_test_late.shape)

print("\nData successfully split for late integration.")


# Apply feature selection with Random Forest Importance for Late Integration datasets
print("\n--- Feature Selection using Random Forest Importance for Late Integration ---")

# Define a function to run RF-based feature selection, as we'll do it for each dataset
def run_rf_selection(X_train, y_train, data_name=""):
    print(f"\nStarting Random Forest feature selection for {data_name} data...")
    # 1. Define the base estimator
    rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    
    # 2. Use SelectFromModel to automatically select features based on median importance
    feature_selector = SelectFromModel(estimator=rf_estimator, threshold='median')
    
    # 3. Fit the selector on the training data
    feature_selector.fit(X_train, y_train)
    
    # 4. Get the boolean mask of selected features
    selected_mask = feature_selector.get_support()
    
    print(f"Random Forest selected {sum(selected_mask)} features from {X_train.shape[1]} for {data_name} data.")
    
    return selected_mask

# Run RF Selection on Microbiome Data
rf_mask_micro = run_rf_selection(X_micro_train, y_train_late, "Microbiome")
X_micro_train_selected = X_micro_train.loc[:, rf_mask_micro]
X_micro_test_selected = X_micro_test.loc[:, rf_mask_micro]

# Run RF Selection on Metabolome Data
rf_mask_metab = run_rf_selection(X_metab_train, y_train_late, "Metabolome")
X_metab_train_selected = X_metab_train.loc[:, rf_mask_metab]
X_metab_test_selected = X_metab_test.loc[:, rf_mask_metab]

print("\n--- Shapes after Random Forest Feature Selection ---")
print("Microbiome Train:", X_micro_train_selected.shape, "| Microbiome Test:", X_micro_test_selected.shape)
print("Metabolome Train:", X_metab_train_selected.shape, "| Metabolome Test:", X_metab_test_selected.shape)


# Strategy 2: Late Integration (Decision-Level) with Feature-Selected Data
print("\n--- Late Integration (Decision-Level) with Feature-Selected Data and GridSearchCV ---")

# We will use the data selected by Boruta. To use RF-selected data, change the dictionaries below.
datasets_train = {'micro': X_micro_train_selected, 'metab': X_metab_train_selected}
datasets_test = {'micro': X_micro_test_selected, 'metab': X_metab_test_selected}

scale_pos_weight_value = (len(y_train_late) - sum(y_train_late)) / sum(y_train_late)

# Define the base models to be used for each dataset
base_models = {
    'LogisticRegression': LogisticRegression(max_iter=1000,
                                             random_state=42),
    'SVC': SVC(probability=True,
               random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42,
                                   verbose=-1),
    'XGBoost': xgb.XGBClassifier(objective='binary:logistic',
                                 booster='gbtree',
                                 tree_method='hist',
                                 random_state=42)
}

param_grids_late = {
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

meta_features_train = []
meta_features_test = []
model_performance = {}

# Set up the cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

# Train base models with GridSearchCV, evaluate them, and generate predictions for the meta-model
for name, model in base_models.items():
    print(f"\n--- Evaluating Base Model: {name} ---")
    for data_type, X_train_curr in datasets_train.items():
        print(f"Tuning and training on {data_type} data...")
        
        X_test_curr = datasets_test[data_type]
        
        # Tune hyperparameters for the base model
        grid_search = GridSearchCV(estimator=model, param_grid=param_grids_late[name], cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=0)
        grid_search.fit(X_train_curr, y_train_late)
        best_model = grid_search.best_estimator_
        
        print(f"  Best params for {name} on {data_type}: {grid_search.best_params_}")
        
        # Fit the best model on the full training data to evaluate it
        best_model.fit(X_train_curr, y_train_late)
        y_proba = best_model.predict_proba(X_test_curr)[:, 1]
        
        # Evaluate and store performance
        auc = roc_auc_score(y_test_late, y_proba)
        model_performance[f"{name}_{data_type}"] = auc
        print(f"  AUC for {name} on {data_type}: {auc:.4f}")

        # Plot Feature Importance for the base model
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4))
        importances = None
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = best_model.coef_[0]

        if importances is not None:
            feature_names = X_train_curr.columns
            base_importances = pd.Series(importances, index=feature_names)
            top_importances = base_importances.abs().nlargest(15)
            
            top_importances.sort_values(ascending=True).plot.barh(ax=ax)
            ax.set_title(f'Top 15 Feature Importances for {name} on {data_type}')
            ax.set_xlabel('Importance')
        else:
            ax.text(0.5, 0.5, 'Feature importances not available', ha='center', va='center')
            ax.set_title(f'Feature Importances for {name} on {data_type}')
            ax.axis('off')
        
        plt.tight_layout()
        fig.savefig(f"{prd}/late_integration_base_{name}_{data_type}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Generate cross-validated predictions for the training set (for meta-model)
        train_preds = cross_val_predict(best_model, X_train_curr, y_train_late, cv=cv_strategy, method='predict_proba')[:, 1]
        meta_features_train.append(train_preds)
        
        # Predict on the test set for the meta-model
        meta_features_test.append(y_proba)

# Identify the best-performing base model
best_base_model_key = max(model_performance, key=model_performance.get)
print(f"\n--- Best Performing Base Model Combination: {best_base_model_key} (AUC: {model_performance[best_base_model_key]:.4f}) ---")

# Combine predictions into final meta-feature matrices
X_late_train = np.array(meta_features_train).T
X_late_test = np.array(meta_features_test).T

print(f"\nShape of the new feature set for the meta-model (train): {X_late_train.shape}")
print(f"Shape of the new feature set for the meta-model (test): {X_late_test.shape}")

# Train and evaluate the final meta-models using all base model types with GridSearchCV
print("\n--- Training and evaluating the final meta-models with GridSearchCV ---")
meta_feature_names = [f"{name}_{data_type}" for name in base_models.keys() for data_type in datasets_train.keys()]

for name, model in base_models.items():
    print(f"\n--- Meta-Model: {name} ---")
    
    # Tune the meta-model
    meta_grid_search = GridSearchCV(estimator=model, param_grid=param_grids_late[name], cv=cv_strategy, scoring='roc_auc', n_jobs=-1, verbose=0)
    meta_grid_search.fit(X_late_train, y_train_late)
    best_meta_model = meta_grid_search.best_estimator_
    
    print(f"  Best params for {name} meta-model: {meta_grid_search.best_params_}")

    y_pred_meta = best_meta_model.predict(X_late_test)
    y_proba_meta = best_meta_model.predict_proba(X_late_test)[:, 1]

    accuracy_meta = accuracy_score(y_test_late, y_pred_meta)
    auc_meta = roc_auc_score(y_test_late, y_proba_meta)

    print("\n--- Performance of Late Integration (Decision-Level) Model ---")
    print(f"  Accuracy: {accuracy_meta:.4f}")
    print(f"  AUC: {auc_meta:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test_late, y_pred_meta, target_names=class_labels, zero_division=0))

    # Plot confusion matrix, AUROC, and Feature Importance for the final meta-model
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4))

    # Confusion Matrix
    cm_meta = confusion_matrix(y_test_late, y_pred_meta)
    sns.heatmap(cm_meta, annot=True, fmt='d', cmap='Greens', xticklabels=class_labels, yticklabels=class_labels, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix for {name} Meta-Model')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test_late, y_proba_meta, ax=axes[1])
    axes[1].set_title(f'ROC Curve for {name} Meta-Model')

    # Feature Importance
    importances = None
    if hasattr(best_meta_model, 'feature_importances_'):
        importances = best_meta_model.feature_importances_
    elif hasattr(best_meta_model, 'coef_'):
        importances = best_meta_model.coef_[0]

    if importances is not None:
        meta_importances = pd.Series(importances, index=meta_feature_names)
        top_importances = meta_importances.abs().nlargest(15)
        
        top_importances.sort_values(ascending=True).plot.barh(ax=axes[2])
        axes[2].set_title('Top 15 Meta-Feature Importances')
        axes[2].set_xlabel('Importance')
    else:
        axes[2].text(0.5, 0.5, 'Feature importances not available', ha='center', va='center')
        axes[2].set_title(f'Feature Importances for {name}')
        axes[2].axis('off')

    plt.tight_layout()
    fig.savefig(f"{prd}/late_integration_meta_{name}_plots.png", dpi=300, bbox_inches='tight')
    plt.close(fig)