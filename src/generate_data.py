import numpy as np
import pandas as pd
from pathlib import Path

def generate_student_data(n_samples=1200, random_seed=42):
    np.random.seed(random_seed)
    
    attendance = np.clip(np.random.normal(75, 15, n_samples), 30, 100)
    internal_marks = np.clip(np.random.normal(55, 18, n_samples), 10, 100)
    assignment_score = np.clip(np.random.normal(65, 15, n_samples), 10, 100)
    study_hours = np.clip(np.random.normal(5, 2.5, n_samples), 0.5, 12)
    prev_performance = np.clip(np.random.normal(60, 18, n_samples), 20, 100)
    
    # Realistic pass/fail rule (not perfectly separable)
    score = (
        0.25 * (attendance / 100) +
        0.30 * (internal_marks / 100) +
        0.15 * (assignment_score / 100) +
        0.15 * (study_hours / 12) +
        0.15 * (prev_performance / 100)
    )
    
    # Add noise to make it non-trivial
    noise = np.random.normal(0, 0.07, n_samples)
    score_noisy = score + noise
    
    # Threshold at ~0.55 so roughly 69% pass, 31% fail
    label = (score_noisy >= 0.55).astype(int)  # 1=Pass, 0=Fail
    
    df = pd.DataFrame({
        'attendance': np.round(attendance, 2),
        'internal_marks': np.round(internal_marks, 2),
        'assignment_score': np.round(assignment_score, 2),
        'study_hours': np.round(study_hours, 2),
        'prev_performance': np.round(prev_performance, 2),
        'result': label
    })
    
    return df

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    df = generate_student_data()
    df.to_csv('data/student_performance.csv', index=False)
    print(f'Dataset generated: {len(df)} records')
    print(df['result'].value_counts())
    print(df.head())
