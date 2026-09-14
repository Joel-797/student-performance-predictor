# Student Performance Predictor

This project predicts a student's exam score from study habits and school information.

## The problem

Exam results are hard to guess in advance. Study time, class attendance, earlier grades, sleep, activities, and parent education all play a part. This project turns those details into a predicted exam score so a student or teacher can see who may need more support before the test.

## What it does

You enter one student's information. A trained machine learning model returns a predicted exam score from 0 to 100 and a pass or fail label. A score of 60 or above counts as a pass. The web form and the command line both use the same saved model. Results come from the trained model, not from hardcoded values.

## Architecture

The project is a small Python pipeline with six parts.

1. Data. `data/students.csv` holds 250 generated students with study features and a final exam score. `src/make_dataset.py` can rebuild that file. `src/config.py` stores paths, feature names, and the pass mark.
2. Loading. `src/data.py` reads the csv file, checks the columns, and splits rows into a training set and a test set.
3. Exploration. `src/explore.py` prints summaries and saves charts in the `reports` folder.
4. Training. `src/train.py` scales numbers, encodes parent education, compares linear regression with a random forest, and saves the best model to `models/performance_model.joblib`. Metrics are written to `reports/metrics.json`.
5. Prediction. `src/predict.py` loads the saved model and scores a new student. Both the command line and the web app call this module.
6. Interface. `app.py` is a Streamlit page where you fill in student fields and receive a live prediction.

Linear regression is the saved model because it had the lower test error on the held out data.

## How to run

Install the packages listed in `requirements.txt` inside a virtual environment.

Rebuild the dataset if you want a fresh csv file:

```
python -m src.make_dataset
```

Explore the data, train the model, then predict from the command line:

```
python -m src.explore
python -m src.train
python -m src.predict
```

The predict command needs hours studied, attendance, previous grade, sleep hours, extracurricular activity (0 or 1), and parent education.

Start the web app:

```
python -m streamlit run app.py
```

Then open http://localhost:8501

In PyCharm, open this folder and set the interpreter to `.venv\Scripts\python.exe`.

## Model results

These figures come from running `python -m src.train` on 200 training rows and 50 test rows.

Linear regression (saved model): test MAE 4.43, test RMSE 5.45, test R2 0.710

Random forest: test MAE 5.26, test RMSE 6.25, test R2 0.618

The full record is in `reports/metrics.json`. Train again after you change the data so the saved model and that file stay in sync.
