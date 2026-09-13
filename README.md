# Student Performance Predictor

Small Python project that predicts a student's final exam score from study habits.

## Open in PyCharm

1. File → Open
2. Choose `C:\Users\user1\PycharmProjects\student-performance-predictor`
3. Use the `.venv` interpreter: Settings → Python Interpreter → Add → Existing → `.venv\Scripts\python.exe`

## Run

```powershell
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.explore
.\.venv\Scripts\python.exe -m src.predict --hours-studied 14 --attendance-pct 90 --previous-grade 78 --sleep-hours 7 --extracurricular 1 --parent-education bachelors
```

## Dataset

`data/students.csv` is a small synthetic sample with:

- hours studied, attendance, previous grade, sleep, extracurriculars, parent education
- target: `final_score` (and a pass/fail flag)
