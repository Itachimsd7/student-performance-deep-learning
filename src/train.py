import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Seeds
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_PATH = Path('data/student_performance.csv')
MODEL_DIR = Path('models')
FIG_DIR = Path('outputs/figures')
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    print('--- Data Info ---')
    print(f'Shape: {df.shape}')
    print(f'Missing values:\n{df.isnull().sum()}')
    print(f'Duplicates: {df.duplicated().sum()}')
    df = df.drop_duplicates()
    print(f'Class distribution:\n{df["result"].value_counts()}')
    
    feature_cols = ['attendance', 'internal_marks', 'assignment_score',
                    'study_hours', 'prev_performance']
    X = df[feature_cols].values
    y = df['result'].values
    
    # One-hot encode labels
    y_cat = to_categorical(y, num_classes=2)
    
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_cat, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=y_temp.argmax(axis=1)
    )
    
    # Scale ONLY fit on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')
    print('\nScaler saved.')
    print(f'Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model():
    model = Sequential([
        Input(shape=(5,)),
        Dense(32, activation='relu', name='hidden_1'),
        Dense(16, activation='relu', name='hidden_2'),
        Dense(2, activation='softmax', name='output')
    ])
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def plot_history(history):
    epochs = range(1, len(history.history['loss']) + 1)
    
    # 1. Training Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['loss'], 'b-o', markersize=3, label='Training Loss')
    plt.title('Training Loss vs Epochs', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'training_loss.png', dpi=150)
    plt.close()
    
    # 2. Validation Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['val_loss'], 'r-o', markersize=3, label='Validation Loss')
    plt.title('Validation Loss vs Epochs', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'validation_loss.png', dpi=150)
    plt.close()
    
    # 3. Training Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['accuracy'], 'g-o', markersize=3, label='Training Accuracy')
    plt.title('Training Accuracy vs Epochs', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'training_accuracy.png', dpi=150)
    plt.close()
    
    # 4. Validation Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.history['val_accuracy'], 'm-o', markersize=3, label='Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs', fontsize=14)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'validation_accuracy.png', dpi=150)
    plt.close()
    
    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history.history['loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history.history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title('Loss vs Epochs'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history.history['accuracy'], 'g-', label='Train Accuracy')
    axes[1].plot(epochs, history.history['val_accuracy'], 'm-', label='Val Accuracy')
    axes[1].set_title('Accuracy vs Epochs'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'training_curves.png', dpi=150)
    plt.close()
    print('Training graphs saved.')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'],
                yticklabels=['Fail', 'Pass'])
    plt.title('Confusion Matrix (Test Set)', fontsize=14)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    print('Confusion matrix saved.')
    return cm

def evaluate(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print('\n--- Test Evaluation ---')
    print(f'Loss:      {loss:.4f}')
    print(f'Accuracy:  {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1-Score:  {f1:.4f}')
    print('\nClassification Report:')
    report_dict = classification_report(y_true, y_pred, target_names=['Fail', 'Pass'], output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=['Fail', 'Pass'])
    
    cm = plot_confusion_matrix(y_true, y_pred)
    
    # Save metrics to JSON
    metrics = {
        'test_loss': float(loss),
        'test_accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report_dict': report_dict,
        'classification_report_text': report_text
    }
    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Metrics saved to outputs/metrics.json')
    return metrics

def main():
    # Generate data if not exists
    if not DATA_PATH.exists():
        import sys
        sys.path.insert(0, 'src')
        from generate_data import generate_student_data
        Path('data').mkdir(exist_ok=True)
        df = generate_student_data()
        df.to_csv(DATA_PATH, index=False)
        print('Data generated.')
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()
    model = build_model()
    
    early_stop = EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    print('\n--- Training ---')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save(MODEL_DIR / 'student_performance_model.keras')
    print('Model saved.')
    
    plot_history(history)
    metrics = evaluate(model, X_test, y_test)
    return metrics

if __name__ == '__main__':
    main()
