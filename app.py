
from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3
from datetime import date, datetime, timedelta

app = Flask(__name__)

# ---------------- LOAD TRAINED MODEL ----------------

saved_data = joblib.load("energy_model.pkl")
model = saved_data["model"]
features = saved_data["features"]
defaults = saved_data["defaults"]

DATABASE = "electricity_readings.db"


# ---------------- DATABASE SETUP ----------------

def get_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meter_id TEXT NOT NULL,
                reading_date TEXT NOT NULL,
                reading_time TEXT NOT NULL,
                meter_reading REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'actual',
                UNIQUE(meter_id, reading_date, reading_time)
            )
        """)


init_database()


# ---------------- WELCOME PAGE ----------------

@app.route("/")
def welcome():
    return render_template("welcome.html")


# ---------------- SAMPLE DEMO ----------------

def create_sample_data():
    yesterday = date.today() - timedelta(days=1)
    today = date.today()

    with get_connection() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO readings
            (meter_id, reading_date, reading_time,
             meter_reading, source)
            VALUES (?, ?, ?, ?, 'sample')
        """, [
            ("SAMPLE-HOUSE", yesterday.isoformat(), "08:00", 1250.0),
            ("SAMPLE-HOUSE", today.isoformat(), "08:00", 1264.0)
        ])

    return calculate_consumption(
        "SAMPLE-HOUSE",
        today.isoformat(),
        "08:00"
    )


# ---------------- CONSUMPTION CALCULATION ----------------

def calculate_consumption(meter_id, reading_date, reading_time):
    current_timestamp = f"{reading_date} {reading_time}"

    with get_connection() as conn:
        current = conn.execute("""
            SELECT *
            FROM readings
            WHERE meter_id = ?
              AND reading_date = ?
              AND reading_time = ?
        """, (meter_id, reading_date, reading_time)).fetchone()

        if current is None:
            return None

        previous = conn.execute("""
            SELECT *
            FROM readings
            WHERE meter_id = ?
              AND (reading_date || ' ' || reading_time) < ?
            ORDER BY reading_date DESC, reading_time DESC
            LIMIT 1
        """, (meter_id, current_timestamp)).fetchone()

    if previous is None:
        return {
            "meter_id": meter_id,
            "date": reading_date,
            "time": reading_time,
            "reading": current["meter_reading"],
            "previous_reading": None,
            "consumption": None,
            "message": (
                "First reading saved. Add an earlier reading "
                "to calculate consumption."
            )
        }

    consumption = (
        float(current["meter_reading"])
        - float(previous["meter_reading"])
    )

    return {
        "meter_id": meter_id,
        "date": reading_date,
        "time": reading_time,
        "reading": current["meter_reading"],
        "previous_reading": previous["meter_reading"],
        "consumption": round(consumption, 2),
        "message": (
            "Consumption calculated using the previous saved reading."
        )
    }


# ---------------- GRAPH DATA ----------------

def get_consumption_graph_data():
    graph_labels = []
    graph_values = []

    with get_connection() as conn:
        recent_readings = conn.execute("""
            SELECT meter_id, reading_date, reading_time, meter_reading
            FROM readings
            ORDER BY reading_date DESC, reading_time DESC, id DESC
            LIMIT 10
        """).fetchall()

        for current in recent_readings:
            timestamp = (
                f"{current['reading_date']} "
                f"{current['reading_time']}"
            )

            previous = conn.execute("""
                SELECT meter_reading
                FROM readings
                WHERE meter_id = ?
                  AND (reading_date || ' ' || reading_time) < ?
                ORDER BY reading_date DESC, reading_time DESC
                LIMIT 1
            """, (current["meter_id"], timestamp)).fetchone()

            if previous is None:
                continue

            consumption = (
                float(current["meter_reading"])
                - float(previous["meter_reading"])
            )

            if consumption < 0:
                continue

            graph_labels.append(
                f"{current['meter_id']} | "
                f"{current['reading_date']} "
                f"{current['reading_time']}"
            )
            graph_values.append(round(consumption, 2))

    graph_labels.reverse()
    graph_values.reverse()

    return graph_labels, graph_values


# ---------------- DASHBOARD ----------------

@app.route("/dashboard", methods=["GET", "POST"])
def home():
    prediction = None
    usage_level = None
    time_period = None
    recommendation = None
    recommendation_title = None
    night_recommendation = None

    tracking_result = None
    tracker_error = None
    tracker_history = []

    if request.method == "POST":
        action = request.form.get("action", "predict")

        # PART 1: ML PREDICTION
        if action == "predict":
            try:
                hour = int(request.form["hour"])
                lights = float(request.form["lights"])
                temperature = float(request.form["temperature"])
                humidity = float(request.form["humidity"])

                if not 0 <= hour <= 23:
                    raise ValueError("Hour must be between 0 and 23.")

                if lights < 0:
                    raise ValueError("Lights cannot be negative.")

                if not 0 <= humidity <= 100:
                    raise ValueError(
                        "Humidity must be between 0 and 100."
                    )

                input_values = defaults.copy()
                input_values["lights"] = lights
                input_values["T1"] = temperature
                input_values["RH_1"] = humidity
                input_values["hour"] = hour

                input_data = pd.DataFrame(
                    [[input_values[name] for name in features]],
                    columns=features
                )

                prediction = round(
                    float(model.predict(input_data)[0]), 2
                )

                time_period = (
                    "Day" if 6 <= hour < 18 else "Night"
                )

                if prediction < 100:
                    usage_level = "Low"
                    recommendation_title = (
                        "🌟 Low Usage — Eco Champion"
                    )
                    recommendation = (
                        "Look at you, Energy Hero! 🦸🌍 "
                        "You're keeping predicted usage low. "
                        "Keep those smart habits going and inspire "
                        "a greener tomorrow!"
                    )

                elif prediction < 200:
                    usage_level = "Medium"
                    recommendation_title = (
                        "⚡ Medium Usage — Tiny Changes, Big Impact"
                    )
                    recommendation = (
                        "You're doing okay! 🌱 A few smart switches "
                        "today could make tomorrow's energy story "
                        "even better. Ready for a little energy-saving "
                        "challenge?"
                    )

                else:
                    usage_level = "High"
                    recommendation_title = (
                        "🔋 High Usage — Energy Detective"
                    )
                    recommendation = (
                        "Oops! Your energy usage is having a power "
                        "party! 🎉⚡ Let's find the energy hogs and "
                        "give those unused appliances a well-deserved "
                        "break!"
                    )

                if time_period == "Night":
                    night_recommendation = (
                        "🌙 Night Mode — Sleep Smart, Save Smart",
                        "The stars are on duty, but your unused "
                        "lights don't need to be! ✨ Switch off what "
                        "you don't need and let your home rest "
                        "smarter tonight."
                    )

            except (ValueError, TypeError, KeyError):
                prediction = None
                usage_level = None
                time_period = None
                recommendation_title = None
                night_recommendation = None
                recommendation = (
                    "Please enter valid values and try again."
                )

        # PART 2: SAVE ACTUAL METER READING
        elif action == "save_reading":
            try:
                meter_id = request.form["meter_id"].strip().upper()
                reading_date = request.form["reading_date"]
                reading_time = request.form["reading_time"]
                meter_reading = float(
                    request.form["meter_reading"]
                )

                if not meter_id or len(meter_id) > 50:
                    raise ValueError("Enter a valid meter ID.")

                date.fromisoformat(reading_date)
                datetime.strptime(reading_time, "%H:%M")

                if date.fromisoformat(reading_date) > date.today():
                    raise ValueError(
                        "Reading date cannot be in the future."
                    )

                if meter_reading < 0:
                    raise ValueError(
                        "Meter reading cannot be negative."
                    )

                timestamp = f"{reading_date} {reading_time}"

                with get_connection() as conn:
                    previous = conn.execute("""
                        SELECT *
                        FROM readings
                        WHERE meter_id = ?
                          AND (reading_date || ' ' || reading_time) < ?
                        ORDER BY reading_date DESC, reading_time DESC
                        LIMIT 1
                    """, (meter_id, timestamp)).fetchone()

                    if (
                        previous
                        and meter_reading < previous["meter_reading"]
                    ):
                        raise ValueError(
                            "This reading is lower than the previous "
                            "reading. Check the meter ID and reading."
                        )

                    conn.execute("""
                        INSERT INTO readings
                        (meter_id, reading_date, reading_time,
                         meter_reading, source)
                        VALUES (?, ?, ?, ?, 'actual')
                    """, (
                        meter_id,
                        reading_date,
                        reading_time,
                        meter_reading
                    ))

                tracking_result = calculate_consumption(
                    meter_id, reading_date, reading_time
                )

            except sqlite3.IntegrityError:
                tracker_error = (
                    "A reading for this meter ID, date and time "
                    "already exists."
                )

            except (ValueError, TypeError):
                tracker_error = (
                    "Please check the meter ID, date, time and reading."
                )

        # PART 3: SAMPLE DEMO
        elif action == "sample_demo":
            try:
                tracking_result = create_sample_data()

                if tracking_result is None:
                    tracker_error = (
                        "Sample reading could not be found. "
                        "Please try again."
                    )

            except sqlite3.Error:
                tracker_error = (
                    "Database error while creating sample data."
                )

    # RECENT SAVED READINGS
    with get_connection() as conn:
        tracker_history = conn.execute("""
            SELECT meter_id, reading_date, reading_time,
                   meter_reading, source
            FROM readings
            ORDER BY reading_date DESC, reading_time DESC, id DESC
            LIMIT 10
        """).fetchall()

    # GRAPH DATA
    graph_labels, graph_values = get_consumption_graph_data()

    return render_template(
        "index.html",
        prediction=prediction,
        usage_level=usage_level,
        time_period=time_period,
        recommendation=recommendation,
        recommendation_title=recommendation_title,
        night_recommendation=night_recommendation,
        tracking_result=tracking_result,
        tracker_error=tracker_error,
        tracker_history=tracker_history,
        graph_labels=graph_labels,
        graph_values=graph_values
    )

# ---------------- ENERGY INSIGHTS PAGE ----------------

@app.route("/insights")
def insights():
    graph_labels, graph_values = get_consumption_graph_data()

    with get_connection() as conn:
        tracker_history = conn.execute("""
            SELECT meter_id, reading_date, reading_time,
                   meter_reading, source
            FROM readings
            ORDER BY reading_date DESC, reading_time DESC, id DESC
            LIMIT 10
        """).fetchall()

    return render_template(
        "insights.html",
        graph_labels=graph_labels,
        graph_values=graph_values,
        tracker_history=tracker_history
    )


# ---------------- THANK YOU PAGE ----------------

@app.route("/thank-you")
def thank_you():
    return render_template("thank_you.html")


if __name__ == "__main__":
    app.run(debug=True)
