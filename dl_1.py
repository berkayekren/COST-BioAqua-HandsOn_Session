# Author: Berkay Ekren
# Date: 2025-10-17
# Description: This script performs multi-omics data integration using deep learning - neural networks.

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import required libraries
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, RocCurveDisplay
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Comment out to make sure only CPU is used
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
plt.savefig('dl_abundance_distribution.png', dpi=300)
plt.close()


print("\n Data Splitting for Deep Learning ")

# We use the raw, unselected features for the DL model.
microbiome_features = microbiome_df.set_index(microbiome_df.columns[0])
metabolome_features = metabolome_df.set_index(metabolome_df.columns[0])

# Transpose the dataframes so that rows are samples and columns are features
X_microbiome = microbiome_features.T
X_metabolome = metabolome_features.T

metadata_indexed = metadata_df.set_index(metadata_df.columns[0])

# Align metadata with the feature dataframe to ensure correct sample order
aligned_metadata = metadata_indexed.reindex(X_microbiome.index)

# Create the classification target from the 'sampling_site' column
y_classification, class_labels = pd.factorize(aligned_metadata[aligned_metadata.columns[1]])

# Step 1: Split into training+validation (80%) and test (20%) sets
X_micro_train_full, X_micro_test, y_train_full, y_test_dl = train_test_split(
    X_microbiome, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)
X_metab_train_full, X_metab_test, _, _ = train_test_split(
    X_metabolome, y_classification, test_size=0.2, random_state=42, stratify=y_classification
)

# Step 2: Split the full training set into a final training set and a validation set
# This results in ~60% train, ~20% validation, 20% test of the original data
X_micro_train, X_micro_val, y_train_dl, y_val_dl = train_test_split(
    X_micro_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)
X_metab_train, X_metab_val, _, _ = train_test_split(
    X_metab_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)


# Display the shapes of the resulting datasets
print("\n--- Shapes of the datasets for Deep Learning ---")
print("Microbiome Data:")
print("Train:", X_micro_train.shape, "| Validation:", X_micro_val.shape, "| Test:", X_micro_test.shape)
print("\nMetabolome Data:")
print("Train:", X_metab_train.shape, "| Validation:", X_metab_val.shape, "| Test:", X_metab_test.shape)
print("\nTarget Variable:")
print("Train:", y_train_dl.shape, "| Validation:", y_val_dl.shape, "| Test:", y_test_dl.shape)

print("\nData successfully split for Deep Learning.")


# Apply feature selection with SelectFromModel
print("\n--- Feature Selection using SelectFromModel for Deep Learning ---")

# Define a function to run the selection
def run_rf_selection(X_train, y_train, data_name=""):
    print(f"\nStarting RF feature selection for {data_name} data...")
    # Using a simple RF classifier to get feature importances
    rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Select features where importance is greater than the median
    # This is a good balance, not too aggressive.
    feature_selector = SelectFromModel(estimator=rf_estimator, threshold='median')
    
    # Fit the selector
    feature_selector.fit(X_train, y_train)
    
    selected_mask = feature_selector.get_support()
    
    print(f"SelectFromModel selected {sum(selected_mask)} features from {X_train.shape[1]} for {data_name} data.")
    
    return selected_mask

# Run Selection on Microbiome Data
rf_mask_micro = run_rf_selection(X_micro_train, y_train_dl, "Microbiome")
X_micro_train_selected = X_micro_train.loc[:, rf_mask_micro]
X_micro_val_selected = X_micro_val.loc[:, rf_mask_micro]
X_micro_test_selected = X_micro_test.loc[:, rf_mask_micro]

# Run Selection on Metabolome Data
rf_mask_metab = run_rf_selection(X_metab_train, y_train_dl, "Metabolome")
X_metab_train_selected = X_metab_train.loc[:, rf_mask_metab]
X_metab_val_selected = X_metab_val.loc[:, rf_mask_metab]
X_metab_test_selected = X_metab_test.loc[:, rf_mask_metab]

print("\n Shapes after SelectFromModel Feature Selection ")
print("Microbiome Train:", X_micro_train_selected.shape, "| Validation:", X_micro_val_selected.shape, "| Test:", X_micro_test_selected.shape)
print("Metabolome Train:", X_metab_train_selected.shape, "| Validation:", X_metab_val_selected.shape, "| Test:", X_metab_test_selected.shape)


##### Feature Extraction with Autoencoders

# Define the dimensionality of the encoded representation
encoding_dim_micro = 32  # Compress microbiome features to 32 dimensions
encoding_dim_metab = 16  # Compress metabolome features to 16 dimensions

# --- Helper function to build an autoencoder ---
def build_autoencoder(input_dim, encoding_dim, regularizer_val=1e-4):
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    # Using a regularizer can help prevent the autoencoder from just learning an identity function
    encoded = Dense(encoding_dim, activation='relu', 
                    activity_regularizer=tf.keras.regularizers.l1(regularizer_val))(input_layer)
    
    # Decoder
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    # Autoencoder model (trains the whole thing)
    autoencoder = Model(input_layer, decoded)
    
    # Encoder model (we'll use this to get the compressed features)
    encoder = Model(input_layer, encoded)
    
    return autoencoder, encoder

# --- Microbiome Autoencoder ---
print("\n--- Training Microbiome Autoencoder ---")
autoencoder_micro, encoder_micro = build_autoencoder(X_micro_train_selected.shape[1], encoding_dim_micro)
autoencoder_micro.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder_micro.fit(X_micro_train_selected, X_micro_train_selected,
                      epochs=100,
                      batch_size=8,
                      shuffle=True,
                      validation_data=(X_micro_val_selected, X_micro_val_selected),
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                      verbose=0) # verbose=0 to keep output clean

# --- Metabolome Autoencoder ---
print("--- Training Metabolome Autoencoder ---")
autoencoder_metab, encoder_metab = build_autoencoder(X_metab_train_selected.shape[1], encoding_dim_metab)
autoencoder_metab.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder_metab.fit(X_metab_train_selected, X_metab_train_selected,
                      epochs=100,
                      batch_size=8,
                      shuffle=True,
                      validation_data=(X_metab_val_selected, X_metab_val_selected),
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                      verbose=0)

# --- Use the Encoders to Transform the Data ---
print("\n--- Encoding the datasets ---")
X_micro_train_encoded = encoder_micro.predict(X_micro_train_selected)
X_micro_val_encoded = encoder_micro.predict(X_micro_val_selected)
X_micro_test_encoded = encoder_micro.predict(X_micro_test_selected)

X_metab_train_encoded = encoder_metab.predict(X_metab_train_selected)
X_metab_val_encoded = encoder_metab.predict(X_metab_val_selected)
X_metab_test_encoded = encoder_metab.predict(X_metab_test_selected)

print("Original microbiome shape:", X_micro_train_selected.shape)
print("Encoded microbiome shape:", X_micro_train_encoded.shape)
print("\nOriginal metabolome shape:", X_metab_train_selected.shape)
print("Encoded metabolome shape:", X_metab_train_encoded.shape)


# Set seeds for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Define the input layers for our NEW encoded data
input_micro_encoded = Input(shape=(encoding_dim_micro,), name='microbiome_encoded_input')
input_metab_encoded = Input(shape=(encoding_dim_metab,), name='metabolome_encoded_input')

# Concatenate the encoded feature branches
combined = Concatenate()([input_micro_encoded, input_metab_encoded])

# Fully Connected Head
# With powerful features, the head can be simpler and more robust.
z = Dense(16, activation='relu')(combined)
z = Dropout(0.5)(z) # A standard dropout rate
output = Dense(1, activation='sigmoid', name='final_output')(z)

# Create and Compile the Model
optimizer = Adam(learning_rate=0.00001)
final_model = Model(inputs=[input_micro_encoded, input_metab_encoded], outputs=output)

final_model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'AUC'])

# Print the model summary
final_model.summary()

# Train the Model on the ENCODED data
history = final_model.fit(
    [X_micro_train_encoded, X_metab_train_encoded],
    y_train_dl,
    epochs=50,
    batch_size=16,
    validation_data=([X_micro_val_encoded, X_metab_val_encoded], y_val_dl),
    callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)],
    verbose=1
)

# Evaluate the Model on the ENCODED Test Set 
print("\n Evaluating the final model on the test set ")
loss, accuracy, auc = final_model.evaluate([X_micro_test_encoded, X_metab_test_encoded], y_test_dl)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")

# Plot Training History 
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.ylim([0, 1])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig('dl_training_history.png', dpi=300)
plt.close()

# --- Generate Final Summary Plots ---
print("\n--- Generating final summary plots ---")

# Get model predictions for the test set
y_proba = final_model.predict([X_micro_test_encoded, X_metab_test_encoded])
y_pred = (y_proba > 0.5).astype("int32")

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Confusion Matrix
cm = confusion_matrix(y_test_dl, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=class_labels, yticklabels=class_labels)
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# 2. ROC Curve
RocCurveDisplay.from_predictions(y_test_dl, y_proba, ax=axes[1])
axes[1].set_title('ROC Curve')

# 3. Encoded Feature Importance
# Get the weights of the first dense layer after concatenation
# The layer index might change if the model architecture changes.
# Based on the current model: Input(0), Input(1), Concat(2), Dense(3)
first_dense_layer_weights = final_model.layers[3].get_weights()[0]

# Sum the absolute weights for each input feature
# This gives a rough measure of importance.
encoded_feature_importance = np.sum(np.abs(first_dense_layer_weights), axis=1)

# Create labels for the encoded features
micro_labels = [f'Micro_E_{i}' for i in range(encoding_dim_micro)]
metab_labels = [f'Metab_E_{i}' for i in range(encoding_dim_metab)]
encoded_feature_labels = micro_labels + metab_labels

# Create a pandas Series for plotting
importances_series = pd.Series(encoded_feature_importance, index=encoded_feature_labels)
top_importances = importances_series.nlargest(15)

top_importances.sort_values().plot(kind='barh', ax=axes[2])
axes[2].set_title('Top 15 Encoded Feature Importances')
axes[2].set_xlabel('Sum of Absolute Weights')

# Save and close the figure
plt.tight_layout()
plt.savefig('dl_summary_plots.png', dpi=300)
plt.close()