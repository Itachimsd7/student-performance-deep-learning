import numpy as np
import pandas as pd
from pathlib import Path

def generate_student_data(n_samples=1500, random_seed=42):
    """
    Generate realistic synthetic university student performance data.
    
    Academic Features:
    - attendance: Class attendance percentage (0 to 100)
    - internal_marks: Internal test/midterm marks (0 to 100)
    - assignment_score: Homework and lab assessments (0 to 100)
    - study_hours: Average daily self-study hours (0 to 12)
    - prev_performance: Previous semester GPA/percentage (0 to 100)
    
    University Academic Rules Incorporated:
    1. Attendance Shortage Cutoff: University mandates minimum 65% attendance.
    2. Internal Passing Minimum: Must score at least 35% in internals.
    3. Composite Performance: Overall academic indicator combining all components.
    4. Non-Separable Noise: Gaussian noise (std=0.04) simulates exam variations.
    """
    np.random.seed(random_seed)
    
    # 1. Generate realistic features spanning the full legitimate range (0-100)
    attendance = np.clip(np.random.normal(72, 20, n_samples), 0, 100)
    internal_marks = np.clip(np.random.normal(54, 22, n_samples), 0, 100)
    assignment_score = np.clip(np.random.normal(62, 20, n_samples), 0, 100)
    study_hours = np.clip(np.random.normal(4.5, 2.8, n_samples), 0, 12)
    prev_performance = np.clip(np.random.normal(58, 20, n_samples), 0, 100)
    
    # 2. Add realistic zero-effort & borderline cohort to ensure the model learns edge boundaries
    # Explicitly ensure ~5% of records have very low/zero attendance or study to ground the model
    low_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    attendance[low_idx] = np.random.uniform(0, 35, len(low_idx))
    study_hours[low_idx] = np.random.uniform(0, 1.5, len(low_idx))
    assignment_score[low_idx] = np.random.uniform(0, 30, len(low_idx))
    
    # 3. Composite Academic Score
    academic_score = (
        0.25 * (attendance / 100.0) +
        0.35 * (internal_marks / 100.0) +
        0.15 * (assignment_score / 100.0) +
        0.10 * (study_hours / 12.0) +
        0.15 * (prev_performance / 100.0)
    )
    
    # 4. Small noise to ensure realistic overlap (not perfectly separable)
    noise = np.random.normal(0, 0.04, n_samples)
    final_score = academic_score + noise
    
    # 5. Strict University Academic Grading Rules:
    # - Must meet minimum attendance cutoff (>= 60%)
    # - Must meet minimum internal passing marks (>= 35%)
    # - Overall composite score must reach passing threshold (>= 0.50)
    labels = (
        (final_score >= 0.50) &
        (attendance >= 60.0) &
        (internal_marks >= 35.0)
    ).astype(int)
    
    df = pd.DataFrame({
        'attendance': np.round(attendance, 2),
        'internal_marks': np.round(internal_marks, 2),
        'assignment_score': np.round(assignment_score, 2),
        'study_hours': np.round(study_hours, 2),
        'prev_performance': np.round(prev_performance, 2),
        'result': labels
    })
    
    return df

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    df = generate_student_data()
    df.to_csv('data/student_performance.csv', index=False)
    print(f'[*] Dataset generated: {len(df)} records')
    print(f'[*] Class Distribution:\n{df["result"].value_counts()}')
    print('\n[*] Sample records:')
    print(df.head())
