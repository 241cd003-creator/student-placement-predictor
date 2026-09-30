from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

app = Flask(__name__)

DATASET = "placement_data.csv"
DATABASE = "placement.db"

FEATURES = [
    "CGPA",
    "AptitudeScore",
    "TechnicalSkills",
    "CommunicationSkills",
    "Internships",
    "Certifications",
    "Projects"
]

TARGET = "PlacementStatus"


# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv(DATASET)

for column in FEATURES:
    data[column] = pd.to_numeric(data[column], errors="coerce")


# ==========================================================
# PREPROCESSING
# ==========================================================

imputer = SimpleImputer(strategy="median")

X = imputer.fit_transform(data[FEATURES])

encoder = LabelEncoder()

y = encoder.fit_transform(
    data[TARGET].astype(str)
)


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

models = {

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}


trained_models = {}

results = {}


for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results[name] = {

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            ),

        "F1 Score":
            f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0
            )
    }

    trained_models[name] = model


# ==========================================================
# SELECT MODEL
# ==========================================================

best_model_name = max(
    results,
    key=lambda x: results[x]["F1 Score"]
)

best_model = trained_models[best_model_name]


# ==========================================================
# POSITIVE CLASS
# ==========================================================

classes = list(encoder.classes_)

positive_index = None

for i, value in enumerate(classes):

    text = str(value).lower().strip()

    if text in [
        "placed",
        "yes",
        "1",
        "true",
        "selected"
    ]:

        positive_index = i
        break


if positive_index is None:

    for i, value in enumerate(classes):

        text = str(value).lower().strip()

        if (
            "placed" in text
            and "not" not in text
            and "un" not in text
        ):

            positive_index = i
            break


if positive_index is None:

    positive_index = len(classes) - 1


# ==========================================================
# DATABASE
# ==========================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            student_name TEXT,

            cgpa REAL,

            aptitude REAL,

            technical REAL,

            communication REAL,

            internships INTEGER,

            certifications INTEGER,

            projects INTEGER,

            prediction TEXT,

            probability REAL,

            model TEXT,

            created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    connection.close()


create_database()


# ==========================================================
# SAVE PREDICTION
# ==========================================================

def save_prediction(
    student_name,
    values,
    prediction,
    probability
):

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (

            student_name,
            cgpa,
            aptitude,
            technical,
            communication,
            internships,
            certifications,
            projects,
            prediction,
            probability,
            model

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        student_name,
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        prediction,
        probability,
        best_model_name
    ))

    connection.commit()

    connection.close()


# ==========================================================
# COMMON HTML HEADER
# ==========================================================

HEADER = """

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1"
>

<title>
Student Placement Prediction System
</title>

<link
href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
rel="stylesheet"
>

<script
src="https://cdn.jsdelivr.net/npm/chart.js"
></script>

<style>

body {
    background: #f5f7fb;
}

.navbar-brand {
    font-weight: 700;
}

.page-title {
    font-weight: 700;
}

.card {
    border: none;
    border-radius: 10px;
}

.metric-card {
    min-height: 120px;
}

select[multiple] {
    height: 150px;
}

.table th {
    white-space: nowrap;
}

</style>

</head>


<body>


<nav class="navbar navbar-expand-lg navbar-dark bg-primary">

<div class="container">

<a
class="navbar-brand"
href="/"
>
Student Placement Prediction System
</a>


<div class="navbar-nav">

<a
class="nav-link"
href="/dataset"
>
Dataset Overview
</a>

<a
class="nav-link"
href="/model"
>
Model Performance
</a>

<a
class="nav-link"
href="/prediction"
>
Prediction
</a>

<a
class="nav-link"
href="/records"
>
Prediction Records
</a>

</div>

</div>

</nav>


<div class="container py-4">

"""


FOOTER = """

</div>

</body>

</html>

"""


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template_string(

        HEADER
        + """

<div class="text-center py-5">

<h1 class="page-title">
Student Placement Prediction System
</h1>

<p class="text-muted mt-3">
AI-based prediction of student placement outcomes
</p>

<a
href="/prediction"
class="btn btn-primary btn-lg mt-3"
>
Start Prediction
</a>

</div>

"""
        + FOOTER
    )


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

@app.route("/dataset")
def dataset_overview():

    total_records = len(data)

    total_features = len(data.columns)

    missing_values = int(
        data.isnull().sum().sum()
    )

    columns = list(data.columns)

    summary = data.describe(
        include="all"
    ).fillna("").round(2)

    table_headers = ""

    for column in summary.columns:

        table_headers += f"""
        <th>{column}</th>
        """

    table_rows = ""

    for index, row in summary.iterrows():

        table_rows += f"<tr><th>{index}</th>"

        for value in row:

            table_rows += f"<td>{value}</td>"

        table_rows += "</tr>"


    target_counts = (
        data[TARGET]
        .astype(str)
        .value_counts()
    )


    chart_labels = list(
        target_counts.index
    )

    chart_values = list(
        target_counts.values
    )


    return render_template_string(

        HEADER
        + """

<h2 class="page-title mb-4">
Dataset Overview
</h2>


<div class="row g-3 mb-4">


<div class="col-md-4">

<div class="card shadow-sm metric-card">

<div class="card-body">

<p class="text-muted mb-1">
Total Records
</p>

<h2>
{{ total_records }}
</h2>

</div>

</div>

</div>


<div class="col-md-4">

<div class="card shadow-sm metric-card">

<div class="card-body">

<p class="text-muted mb-1">
Total Features
</p>

<h2>
{{ total_features }}
</h2>

</div>

</div>

</div>


<div class="col-md-4">

<div class="card shadow-sm metric-card">

<div class="card-body">

<p class="text-muted mb-1">
Missing Values
</p>

<h2>
{{ missing_values }}
</h2>

</div>

</div>

</div>


</div>


<div class="card shadow-sm mb-4">

<div class="card-body">

<h5 class="fw-bold mb-3">
Dataset Columns
</h5>

<div class="table-responsive">

<table class="table table-bordered">

<thead class="table-light">

<tr>

{% for column in columns %}

<th>{{ column }}</th>

{% endfor %}

</tr>

</thead>

</table>

</div>

</div>

</div>


<div class="row g-4">


<div class="col-lg-7">

<div class="card shadow-sm">

<div class="card-body">

<h5 class="fw-bold mb-3">
Statistical Summary
</h5>

<div class="table-responsive">

<table class="table table-bordered table-sm">

<thead class="table-light">

<tr>

<th>Statistic</th>

{{ headers|safe }}

</tr>

</thead>

<tbody>

{{ rows|safe }}

</tbody>

</table>

</div>

</div>

</div>

</div>


<div class="col-lg-5">

<div class="card shadow-sm">

<div class="card-body">

<h5 class="fw-bold mb-3">
Placement Distribution
</h5>

<canvas id="placementChart"></canvas>

</div>

</div>

</div>


</div>


<script>

new Chart(

document.getElementById(
"placementChart"
),

{

type: "pie",

data: {

labels:
{{ chart_labels|safe }},

datasets: [{

data:
{{ chart_values|safe }}

}]

},

options: {

responsive: true

}

}

);

</script>

"""
        + FOOTER,

        total_records=total_records,

        total_features=total_features,

        missing_values=missing_values,

        columns=columns,

        headers=table_headers,

        rows=table_rows,

        chart_labels=chart_labels,

        chart_values=chart_values
    )


# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

@app.route("/model")
def model_performance():

    table_rows = ""

    accuracy_values = []

    precision_values = []

    recall_values = []

    f1_values = []

    model_names = []


    for name, metric in results.items():

        model_names.append(name)

        accuracy_values.append(
            round(
                metric["Accuracy"] * 100,
                2
            )
        )

        precision_values.append(
            round(
                metric["Precision"] * 100,
                2
            )
        )

        recall_values.append(
            round(
                metric["Recall"] * 100,
                2
            )
        )

        f1_values.append(
            round(
                metric["F1 Score"] * 100,
                2
            )
        )


        table_rows += f"""

<tr>

<td>
{name}
</td>

<td>
{metric["Accuracy"]:.2%}
</td>

<td>
{metric["Precision"]:.2%}
</td>

<td>
{metric["Recall"]:.2%}
</td>

<td>
{metric["F1 Score"]:.2%}
</td>

</tr>

"""


    return render_template_string(

        HEADER
        + """

<h2 class="page-title mb-4">
Model Performance
</h2>


<div class="card shadow-sm mb-4">

<div class="card-body">

<h5 class="fw-bold mb-3">
Classification Model Comparison
</h5>


<div class="table-responsive">

<table class="table table-bordered">

<thead class="table-light">

<tr>

<th>Model</th>

<th>Accuracy</th>

<th>Precision</th>

<th>Recall</th>

<th>F1 Score</th>

</tr>

</thead>


<tbody>

{{ table_rows|safe }}

</tbody>

</table>

</div>

</div>

</div>


<div class="card shadow-sm mb-4">

<div class="card-body">

<h5 class="fw-bold mb-3">
Performance Visualization
</h5>

<canvas id="modelChart"></canvas>

</div>

</div>


<div class="card shadow-sm">

<div class="card-body">

<h5 class="fw-bold">
Selected Model
</h5>

<p class="mb-0">
{{ best_model }}
</p>

</div>

</div>


<script>

new Chart(

document.getElementById(
"modelChart"
),

{

type: "bar",

data: {

labels:
{{ model_names|safe }},

datasets: [

{

label: "Accuracy",

data:
{{ accuracy_values|safe }}

},

{

label: "Precision",

data:
{{ precision_values|safe }}

},

{

label: "Recall",

data:
{{ recall_values|safe }}

},

{

label: "F1 Score",

data:
{{ f1_values|safe }}

}

]

},

options: {

responsive: true,

scales: {

y: {

beginAtZero: true,

max: 100

}

}

}

}

);

</script>

"""
        + FOOTER,

        table_rows=table_rows,

        best_model=best_model_name,

        model_names=model_names,

        accuracy_values=accuracy_values,

        precision_values=precision_values,

        recall_values=recall_values,

        f1_values=f1_values
    )


# ==========================================================
# PREDICTION PAGE
# ==========================================================

@app.route("/prediction")
def prediction_page():

    return render_template_string(

        HEADER
        + """

<h2 class="page-title mb-4">
Student Placement Prediction
</h2>


<div class="card shadow-sm">

<div class="card-body p-4">


<form
method="POST"
action="/predict"
>


<div class="row g-3">


<div class="col-md-6">

<label class="form-label">
Student Name
</label>

<input
type="text"
name="student_name"
class="form-control"
required
>

</div>


<div class="col-md-6">

<label class="form-label">
CGPA
</label>

<input
type="number"
name="CGPA"
class="form-control"
min="0"
max="10"
step="0.01"
required
>

</div>


<div class="col-md-6">

<label class="form-label">
Aptitude Score
</label>

<input
type="number"
name="AptitudeScore"
class="form-control"
min="0"
max="100"
required
>

</div>


<div class="col-md-6">

<label class="form-label">
Internships
</label>

<input
type="number"
name="Internships"
class="form-control"
min="0"
required
>

</div>


<div class="col-md-6">

<label class="form-label">
Projects
</label>

<input
type="number"
name="Projects"
class="form-control"
min="0"
required
>

</div>


<div class="col-md-6">

<label class="form-label">
Technical Skills
</label>

<select
name="TechnicalSkills"
class="form-select"
multiple
required
>

<option value="Python">
Python
</option>

<option value="Java">
Java
</option>

<option value="C++">
C++
</option>

<option value="SQL">
SQL
</option>

<option value="Machine Learning">
Machine Learning
</option>

<option value="Data Science">
Data Science
</option>

<option value="Web Development">
Web Development
</option>

<option value="Cloud Computing">
Cloud Computing
</option>

<option value="Power BI">
Power BI
</option>

<option value="Excel">
Excel
</option>

</select>

<small class="text-muted">
Hold Ctrl to select multiple skills
</small>

</div>


<div class="col-md-6">

<label class="form-label">
Communication Skills
</label>

<select
name="CommunicationSkills"
class="form-select"
multiple
required
>

<option value="English Speaking">
English Speaking
</option>

<option value="Presentation">
Presentation
</option>

<option value="Teamwork">
Teamwork
</option>

<option value="Leadership">
Leadership
</option>

<option value="Public Speaking">
Public Speaking
</option>

<option value="Interview Skills">
Interview Skills
</option>

<option value="Active Listening">
Active Listening
</option>

<option value="Written Communication">
Written Communication
</option>

</select>

<small class="text-muted">
Hold Ctrl to select multiple skills
</small>

</div>


<div class="col-md-6">

<label class="form-label">
Certifications
</label>

<select
name="Certifications"
class="form-select"
multiple
required
>

<option value="Python Certification">
Python Certification
</option>

<option value="Java Certification">
Java Certification
</option>

<option value="SQL Certification">
SQL Certification
</option>

<option value="Machine Learning Certification">
Machine Learning Certification
</option>

<option value="Data Analytics Certification">
Data Analytics Certification
</option>

<option value="Cloud Certification">
Cloud Certification
</option>

<option value="Power BI Certification">
Power BI Certification
</option>

<option value="Web Development Certification">
Web Development Certification
</option>

</select>

<small class="text-muted">
Hold Ctrl to select multiple certifications
</small>

</div>


</div>


<div class="text-center mt-4">

<button
type="submit"
class="btn btn-primary px-5"
>
Predict Placement
</button>

</div>


</form>


</div>

</div>

"""
        + FOOTER
    )


# ==========================================================
# PREDICTION PROCESS
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        student_name = request.form[
            "student_name"
        ]


        cgpa = float(
            request.form["CGPA"]
        )


        aptitude = float(
            request.form[
                "AptitudeScore"
            ]
        )


        internships = int(
            request.form[
                "Internships"
            ]
        )


        projects = int(
            request.form[
                "Projects"
            ]
        )


        technical_skills = request.form.getlist(
            "TechnicalSkills"
        )


        communication_skills = request.form.getlist(
            "CommunicationSkills"
        )


        certifications = request.form.getlist(
            "Certifications"
        )


        technical_score = min(
            len(technical_skills) * 10,
            100
        )


        communication_score = min(
            len(communication_skills) * 10,
            100
        )


        certification_count = len(
            certifications
        )


        values = [

            cgpa,

            aptitude,

            technical_score,

            communication_score,

            internships,

            certification_count,

            projects

        ]


        input_data = imputer.transform(

            pd.DataFrame(
                [values],
                columns=FEATURES
            )

        )


        prediction = best_model.predict(
            input_data
        )[0]


        probabilities = best_model.predict_proba(
            input_data
        )[0]


        model_classes = list(
            best_model.classes_
        )


        positive_probability_index = (
            model_classes.index(
                positive_index
            )
        )


        probability = (

            probabilities[
                positive_probability_index
            ]

            * 100

        )


        if prediction == positive_index:

            status = "Likely to be Placed"

        else:

            status = "Unlikely to be Placed"


        save_prediction(

            student_name,

            values,

            status,

            probability

        )


        return render_template_string(

            HEADER
            + """

<div class="card shadow-sm mx-auto"
style="max-width:700px;">

<div class="card-body p-5 text-center">


<h2 class="fw-bold">
Prediction Result
</h2>


<p class="text-muted">
Student: {{ name }}
</p>


<div class="alert
{% if status == 'Likely to be Placed' %}
alert-success
{% else %}
alert-danger
{% endif %}
mt-4">

<h4 class="fw-bold">
{{ status }}
</h4>

<h5 class="mt-3">

Placement Probability:
{{ "%.2f"|format(probability) }}%

</h5>

</div>


<div class="text-start mt-4">


<p>
<strong>Technical Skills:</strong>
{{ technical_count }}
selected
</p>


<p>
<strong>Communication Skills:</strong>
{{ communication_count }}
selected
</p>


<p>
<strong>Certifications:</strong>
{{ certification_count }}
selected
</p>


<p>
<strong>Model Used:</strong>
{{ model }}
</p>


</div>


<div class="mt-4">

<a
href="/prediction"
class="btn btn-primary"
>
New Prediction
</a>


<a
href="/records"
class="btn btn-outline-secondary ms-2"
>
View Records
</a>

</div>


</div>

</div>

"""
            + FOOTER,

            name=student_name,

            status=status,

            probability=probability,

            technical_count=len(
                technical_skills
            ),

            communication_count=len(
                communication_skills
            ),

            certification_count=len(
                certifications
            ),

            model=best_model_name
        )


    except Exception as error:

        return render_template_string(

            HEADER
            + """

<div class="alert alert-danger">

<h5>
Prediction Error
</h5>

<p>
{{ error }}
</p>

<a href="/prediction">
Go Back
</a>

</div>

"""
            + FOOTER,

            error=str(error)
        )


# ==========================================================
# PREDICTION RECORDS
# ==========================================================

@app.route("/records")
def prediction_records():

    connection = sqlite3.connect(
        DATABASE
    )


    records = pd.read_sql_query(

        """
        SELECT
            id,
            student_name,
            prediction,
            probability,
            model,
            created_at
        FROM predictions
        ORDER BY id DESC
        """,

        connection
    )


    connection.close()


    rows = ""


    if records.empty:

        rows = """

<tr>

<td
colspan="6"
class="text-center"
>

No prediction records found.

</td>

</tr>

"""


    else:

        for _, row in records.iterrows():

            rows += f"""

<tr>

<td>
{row["id"]}
</td>

<td>
{row["student_name"]}
</td>

<td>
{row["prediction"]}
</td>

<td>
{row["probability"]:.2f}%
</td>

<td>
{row["model"]}
</td>

<td>
{row["created_at"]}
</td>

</tr>

"""


    return render_template_string(

        HEADER
        + """

<h2 class="page-title mb-4">
Prediction Records
</h2>


<div class="card shadow-sm">

<div class="card-body">


<div class="table-responsive">

<table class="table table-bordered table-hover">

<thead class="table-light">

<tr>

<th>ID</th>

<th>Student Name</th>

<th>Prediction</th>

<th>Probability</th>

<th>Model</th>

<th>Date & Time</th>

</tr>

</thead>


<tbody>

{{ rows|safe }}

</tbody>


</table>

</div>


</div>

</div>

"""
        + FOOTER,

        rows=rows
    )


# ==========================================================
# REST API
# ==========================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    try:

        request_data = request.get_json()


        values = [

            float(
                request_data["CGPA"]
            ),

            float(
                request_data[
                    "AptitudeScore"
                ]
            ),

            float(
                request_data[
                    "TechnicalSkills"
                ]
            ),

            float(
                request_data[
                    "CommunicationSkills"
                ]
            ),

            float(
                request_data[
                    "Internships"
                ]
            ),

            float(
                request_data[
                    "Certifications"
                ]
            ),

            float(
                request_data[
                    "Projects"
                ]
            )

        ]


        input_data = imputer.transform(

            pd.DataFrame(
                [values],
                columns=FEATURES
            )

        )


        prediction = best_model.predict(
            input_data
        )[0]


        probabilities = best_model.predict_proba(
            input_data
        )[0]


        model_classes = list(
            best_model.classes_
        )


        probability_index = (
            model_classes.index(
                positive_index
            )
        )


        probability = (

            probabilities[
                probability_index
            ]

            * 100

        )


        if prediction == positive_index:

            status = "Likely to be Placed"

        else:

            status = "Unlikely to be Placed"


        return jsonify({

            "prediction":
                status,

            "placement_probability":
                round(
                    probability,
                    2
                ),

            "model":
                best_model_name

        })


    except Exception as error:

        return jsonify({

            "error":
                str(error)

        }), 400


# ==========================================================
# MODEL METRICS API
# ==========================================================

@app.route(
    "/api/metrics"
)
def api_metrics():

    return jsonify(
        results
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )