
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Student Pass Prediction API")

MODEL_PATH = Path(__file__).with_name("student_model.joblib")
model = joblib.load(MODEL_PATH)


class Student(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    previous_marks: float = Field(ge=0, le=100)
    attendance: float = Field(ge=0, le=100)
    assignments_completed: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0, le=12)


@app.get("/")
def home():
    return {"message": "Student ML API is running"}


@app.post("/predict")
def predict(student: Student):
    data = pd.DataFrame([{
        "study_hours": student.study_hours,
        "previous_marks": student.previous_marks,
        "attendance": student.attendance,
        "assignments_completed": student.assignments_completed,
        "sleep_hours": student.sleep_hours,
    }])

    prediction = int(model.predict(data)[0])

    return {
        "prediction": prediction,
        "result": "Pass" if prediction == 1 else "Fail"
    }