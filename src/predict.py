import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path
import argparse

MODEL_PATH = Path('models/student_performance_model.keras')
SCALER_PATH = Path('models/scaler.pkl')

_model = None
_scaler = None

def get_model_and_scaler():
    """Lazy load model and scaler with caching."""
    global _model, _scaler
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _scaler is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}. Train the model first.")
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler

def predict_student(attendance: float,
                    internal_marks: float,
                    assignment_score: float,
                    study_hours: float,
                    prev_performance: float):
    """
    Predict whether a student will PASS or FAIL given 5 input features.

    Parameters:
    - attendance: Percentage (0-100)
    - internal_marks: Marks (0-100)
    - assignment_score: Marks (0-100)
    - study_hours: Daily study hours (0-12)
    - prev_performance: Percentage/Marks (0-100)

    Returns:
    dict containing:
    - prediction: 'PASS' or 'FAIL'
    - pass_probability: float (0.0 to 1.0)
    - fail_probability: float (0.0 to 1.0)
    - input_features: dict of supplied values
    """
    model, scaler = get_model_and_scaler()

    # Feature order must match training
    raw_features = np.array([[attendance, internal_marks, assignment_score, study_hours, prev_performance]], dtype=float)
    scaled_features = scaler.transform(raw_features)

    probabilities = model.predict(scaled_features, verbose=0)[0]
    fail_prob = float(probabilities[0])
    pass_prob = float(probabilities[1])

    result = "PASS" if pass_prob >= fail_prob else "FAIL"

    return {
        "prediction": result,
        "pass_probability": pass_prob,
        "fail_probability": fail_prob,
        "input_features": {
            "attendance": attendance,
            "internal_marks": internal_marks,
            "assignment_score": assignment_score,
            "study_hours": study_hours,
            "prev_performance": prev_performance
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Student Pass/Fail outcome.")
    parser.add_argument("--attendance", type=float, default=80.0, help="Attendance percentage (0-100)")
    parser.add_argument("--internal", type=float, default=65.0, help="Internal marks (0-100)")
    parser.add_argument("--assignment", type=float, default=70.0, help="Assignment score (0-100)")
    parser.add_argument("--hours", type=float, default=5.0, help="Daily study hours (0-12)")
    parser.add_argument("--previous", type=float, default=65.0, help="Previous academic performance (0-100)")

    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("[*] STUDENT PERFORMANCE PREDICTOR")
    print("="*50)
    
    res = predict_student(args.attendance, args.internal, args.assignment, args.hours, args.previous)
    
    print(f"Inputs:")
    for k, v in res['input_features'].items():
        print(f"  - {k:20s}: {v}")
    print("-"*50)
    print(f"Result:           {res['prediction']}")
    print(f"Pass Probability: {res['pass_probability'] * 100:.2f}%")
    print(f"Fail Probability: {res['fail_probability'] * 100:.2f}%")
    print("="*50 + "\n")
