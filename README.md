# Student Performance Prediction
This project uses machine learning to predict student performance based on various features (demographics, grades, etc.).

## Project Structure

- `data/` — datasets and instructions
- `notebooks/` — Jupyter notebooks for exploration and prototyping
- `src/` — scripts for preprocessing, training, and prediction
- `results/` — model results and metrics

## How to Run

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Download the dataset (see `data/README.md`).
3. Run the notebook or scripts in `src/`.

## Model

We use a Random Forest classifier to predict whether a student will pass or fail.

## Results

See `results/model_metrics.txt` for evaluation metrics.
