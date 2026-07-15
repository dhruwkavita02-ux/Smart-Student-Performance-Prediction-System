from flask import Flask, render_template, request,redirect
import sqlite3
import joblib
import numpy as np

# Load the trained model and preprocessing objects
app = Flask(__name__)
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")

def create_table():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students(
        student_id TEXT PRIMARY KEY,
        name TEXT,
        age INTEGER,
        gender TEXT,
        branch TEXT,
        semester INTEGER,
        attendance REAL,
        internal_marks REAL,
        assignment_marks REAL,
        study_hours REAL,
        previous_percentage REAL,
        activities TEXT
    )
    """)

    conn.commit()
    conn.close()

create_table()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/dashboard',methods=['POST'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():

    student_id = request.form['student_id']
    student_name = request.form['student_name']
    age = request.form['age']
    gender = request.form['gender']
    branch = request.form['branch']
    semester = request.form['semester']
    attendance = request.form['attendance']
    internal_marks = request.form['internal_marks']
    assignment = request.form['assignment']
    study_hours = request.form['study_hours']
    previous_percentage = request.form['previous_percentage']
    activities = request.form['activities']

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO students
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
    student_id,
    student_name,
    age,
    gender,
    branch,
    semester,
    attendance,
    internal_marks,
    assignment,
    study_hours,
    previous_percentage,
    activities
))

    conn.commit()
    conn.close()

    input_data = np.array([[
        float(attendance),
        float(internal_marks),
        float(assignment),
        float(study_hours),
        float(previous_percentage)
    ]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    prediction = encoder.inverse_transform(prediction)

    if prediction[0] == "Excellent":
     suggestion = "Excellent performance! Keep maintaining your consistency."

    elif prediction[0] == "Good":
     suggestion = "Good performance. Improve your study hours to reach Excellent."

    elif prediction[0] == "Average":
     suggestion = "Increase attendance, complete assignments regularly and study daily."

    else:
     suggestion = "Needs improvement. Focus on attendance, study hours and internal marks."

    return render_template(
    "result.html",
    prediction=prediction[0],
    suggestion=suggestion
)
@app.route('/records')
def records():

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students")

    students = cursor.fetchall()

    conn.close()

    return render_template("records.html", students=students)
@app.route('/delete/<student_id>')
def delete(student_id):

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM students WHERE student_id=?",
        (student_id,)
    )

    conn.commit()
    conn.close()

    return redirect("/records")

@app.route('/report')
def report():
    return "<h2>Report Page (Coming Soon)</h2>"

if __name__ == '__main__':
    app.run(debug=True)