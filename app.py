
from flask import Flask, render_template, request
import sqlite3
import math
from datetime import date, datetime, timedelta
from statistics import mean
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
DATABASE = "electricity_readings.db"


# ---------------- DATABASE ----------------

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


# ---------------- METER CONSUMPTION ----------------

def calculate_consumption(meter_id, reading_date, reading_time):

    timestamp = f"{reading_date} {reading_time}"

    with get_connection() as conn:
        current = conn.execute("""
            SELECT *
            FROM readings
            WHERE meter_id = ?
              AND reading_date = ?
              AND reading_time = ?
            LIMIT 1
        """, (meter_id, reading_date, reading_time)).fetchone()

        if current is None:
            return None

        previous = conn.execute("""
            SELECT *
            FROM readings
            WHERE meter_id = ?
              AND (reading_date || ' ' || reading_time) < ?
            ORDER BY reading_date DESC, reading_time DESC, id DESC
            LIMIT 1
        """, (meter_id, timestamp)).fetchone()

    if previous is None:
        return {
            "meter_id": meter_id,
            "date": reading_date,
            "time": reading_time,
            "reading": current["meter_reading"],
            "previous_reading": None,
            "consumption": None,
            "message": "First reading saved. Add another reading to calculate consumption."
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
        "message": "Consumption calculated successfully."
    }


# ---------------- DAILY CONSUMPTION HISTORY ----------------

def get_daily_consumptions(meter_id):

    with get_connection() as conn:
        readings = conn.execute("""
            SELECT reading_date, reading_time, meter_reading
            FROM readings
            WHERE meter_id = ?
            ORDER BY reading_date ASC, reading_time ASC, id ASC
        """, (meter_id,)).fetchall()

    daily_readings = {}

    # Use the last reading recorded on each date.
    for reading in readings:
        daily_readings[reading["reading_date"]] = float(
            reading["meter_reading"]
        )

    dates = sorted(daily_readings.keys())
    daily_consumptions = []

    for i in range(1, len(dates)):

        previous_date = date.fromisoformat(dates[i - 1])
        current_date = date.fromisoformat(dates[i])

        # Calculate daily consumption only for consecutive dates.
        if (current_date - previous_date).days != 1:
            continue

        usage = (
            daily_readings[dates[i]]
            - daily_readings[dates[i - 1]]
        )

        if usage >= 0:
            daily_consumptions.append(
                (current_date, round(usage, 2))
            )

    return daily_consumptions


# ---------------- NEXT-DAY ML PREDICTION ----------------

def predict_next_day(meter_id):

    history = get_daily_consumptions(meter_id)

    if len(history) < 8:
        return {
            "prediction": None,
            "level": None,
            "recommendation": None,
            "message": (
                "Save at least 8 consecutive daily consumption values. "
                "Usually 9 daily meter readings are required for prediction."
            )
        }

    values = [item[1] for item in history]

    X = []
    y = []

    for i in range(3, len(values)):

        previous_values = values[max(0, i - 7):i]

        features = [
            values[i - 1],
            values[i - 2],
            values[i - 3],
            mean(previous_values),
            history[i][0].weekday()
        ]

        X.append(features)
        y.append(values[i])

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    latest_values = values[-7:]
    tomorrow = date.today() + timedelta(days=1)

    input_data = [[
        values[-1],
        values[-2],
        values[-3],
        mean(latest_values),
        tomorrow.weekday()
    ]]

    prediction = float(model.predict(input_data)[0])
    prediction = max(0, round(prediction, 2))

    average_usage = mean(latest_values)

    if prediction < average_usage * 0.8:
        level = "Low"
        recommendation = (
            "Your predicted electricity usage is low. "
            "Continue your energy-saving habits and switch off unused devices."
        )

    elif prediction > average_usage * 1.2:
        level = "High"
        recommendation = (
            "Your predicted electricity usage is high. "
            "Avoid unnecessary electricity usage and switch off unused appliances."
        )

    else:
        level = "Medium"
        recommendation = (
            "Your predicted electricity usage is moderate. "
            "Try to reduce unnecessary usage and follow energy-saving habits."
        )

    return {
        "prediction": prediction,
        "level": level,
        "recommendation": recommendation,
        "message": (
            f"Prediction generated using {len(values)} daily "
            "consumption records."
        )
    }


# ---------------- YESTERDAY VS TODAY COMPARISON ----------------

def get_day_consumption(meter_id, target_date):

    previous_date = target_date - timedelta(days=1)

    with get_connection() as conn:

        current = conn.execute("""
            SELECT meter_reading
            FROM readings
            WHERE meter_id = ?
              AND reading_date = ?
            ORDER BY reading_time DESC, id DESC
            LIMIT 1
        """, (meter_id, target_date.isoformat())).fetchone()

        previous = conn.execute("""
            SELECT meter_reading
            FROM readings
            WHERE meter_id = ?
              AND reading_date = ?
            ORDER BY reading_time DESC, id DESC
            LIMIT 1
        """, (meter_id, previous_date.isoformat())).fetchone()

    if current is None or previous is None:
        return None

    usage = (
        float(current["meter_reading"])
        - float(previous["meter_reading"])
    )

    if usage < 0:
        return None

    return round(usage, 2)


def get_daily_comparison(meter_id):

    today = date.today()
    yesterday = today - timedelta(days=1)

    today_usage = get_day_consumption(meter_id, today)
    yesterday_usage = get_day_consumption(meter_id, yesterday)

    result = {
        "today": today_usage,
        "yesterday": yesterday_usage,
        "difference": None,
        "status": "Waiting",
        "message": (
            "Comparison requires readings for today, yesterday "
            "and the day before yesterday."
        ),
        "recommendation": (
            "Save daily meter readings to compare your electricity usage."
        )
    }

    if today_usage is None or yesterday_usage is None:
        return result

    difference = round(today_usage - yesterday_usage, 2)
    result["difference"] = difference

    if difference > 0:
        result["status"] = "Increased"
        result["message"] = (
            f"Electricity usage increased by {difference} units compared to yesterday."
        )
        result["recommendation"] = (
            "Reduce unnecessary electricity usage. Switch off unused lights "
            "and electrical appliances."
        )

    elif difference < 0:
        result["status"] = "Decreased"
        result["message"] = (
            f"Great! Electricity usage decreased by {abs(difference)} units compared to yesterday."
        )
        result["recommendation"] = (
            "Good progress! Continue following your energy-saving habits."
        )

    else:
        result["status"] = "Same"
        result["message"] = (
            "Your electricity usage is the same as yesterday."
        )
        result["recommendation"] = (
            "Try to reduce unnecessary electricity usage to save more energy."
        )

    return result


# ---------------- SAMPLE DEMO DATA ----------------

def create_sample_data():

    meter_id = "DEMO-ML-001"

    daily_usage = [
        12, 14, 13, 15, 16,
        14, 17, 15, 18, 16,
        14, 17, 19, 16
    ]

    start_date = date.today() - timedelta(days=14)
    meter_reading = 1000.0

    with get_connection() as conn:

        conn.execute("""
            INSERT OR IGNORE INTO readings
            (meter_id, reading_date, reading_time, meter_reading, source)
            VALUES (?, ?, ?, ?, 'sample')
        """, (
            meter_id,
            start_date.isoformat(),
            "08:00",
            meter_reading
        ))

        for i, usage in enumerate(daily_usage):

            meter_reading += usage

            reading_date = (
                start_date + timedelta(days=i + 1)
            ).isoformat()

            conn.execute("""
                INSERT OR IGNORE INTO readings
                (meter_id, reading_date, reading_time, meter_reading, source)
                VALUES (?, ?, ?, ?, 'sample')
            """, (
                meter_id,
                reading_date,
                "08:00",
                meter_reading
            ))

    return calculate_consumption(
        meter_id,
        date.today().isoformat(),
        "08:00"
    )


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
                ORDER BY reading_date DESC, reading_time DESC, id DESC
                LIMIT 1
            """, (
                current["meter_id"],
                timestamp
            )).fetchone()

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
                f"{current['reading_date']} {current['reading_time']}"
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

    next_day_prediction = None
    prediction_level = None
    prediction_recommendation = None
    prediction_message = None

    comparison_today = None
    comparison_yesterday = None
    comparison_difference = None
    comparison_status = None
    comparison_message = None
    comparison_recommendation = None

    tracking_result = None
    tracker_error = None
    tracker_history = []

    selected_meter_id = None

    if request.method == "POST":

        action = request.form.get("action", "")

        # SAVE ACTUAL METER READING
        if action == "save_reading":

            try:
                meter_id = request.form.get(
                    "meter_id", ""
                ).strip().upper()

                reading_date = request.form.get("reading_date", "")
                reading_time = request.form.get("reading_time", "")

                meter_reading = float(
                    request.form.get("meter_reading", "")
                )

                if not meter_id or len(meter_id) > 50:
                    raise ValueError("Enter a valid meter ID.")

                parsed_date = date.fromisoformat(reading_date)
                datetime.strptime(reading_time, "%H:%M")

                if parsed_date > date.today():
                    raise ValueError("Reading date cannot be in the future.")

                if not math.isfinite(meter_reading) or meter_reading < 0:
                    raise ValueError("Enter a valid non-negative meter reading.")

                timestamp = f"{reading_date} {reading_time}"

                with get_connection() as conn:

                    previous = conn.execute("""
                        SELECT *
                        FROM readings
                        WHERE meter_id = ?
                          AND (reading_date || ' ' || reading_time) < ?
                        ORDER BY reading_date DESC, reading_time DESC, id DESC
                        LIMIT 1
                    """, (
                        meter_id,
                        timestamp
                    )).fetchone()

                    if (
                        previous
                        and meter_reading < float(previous["meter_reading"])
                    ):
                        raise ValueError(
                            "This reading is lower than the previous reading. "
                            "Check the meter ID and reading."
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

                selected_meter_id = meter_id

                tracking_result = calculate_consumption(
                    meter_id,
                    reading_date,
                    reading_time
                )

            except sqlite3.IntegrityError:
                tracker_error = (
                    "A reading for this meter ID, date and time already exists."
                )

            except (ValueError, TypeError):
                tracker_error = (
                    "Please check the meter ID, date, time and reading."
                )

        # SAMPLE DEMO
        elif action == "sample_demo":

            try:
                tracking_result = create_sample_data()
                selected_meter_id = "DEMO-ML-001"

                if tracking_result is None:
                    tracker_error = "Sample reading could not be found."

            except sqlite3.Error:
                tracker_error = (
                    "Database error while creating sample data."
                )

    # RECENT READINGS AND SELECTED METER
    with get_connection() as conn:

        tracker_history = conn.execute("""
            SELECT meter_id, reading_date, reading_time,
                   meter_reading, source
            FROM readings
            ORDER BY reading_date DESC, reading_time DESC, id DESC
            LIMIT 10
        """).fetchall()

        if selected_meter_id is None:

            latest = conn.execute("""
                SELECT meter_id
                FROM readings
                ORDER BY reading_date DESC, reading_time DESC, id DESC
                LIMIT 1
            """).fetchone()

            if latest:
                selected_meter_id = latest["meter_id"]

    # NEXT-DAY PREDICTION
    if selected_meter_id:

        result = predict_next_day(selected_meter_id)

        next_day_prediction = result["prediction"]
        prediction_level = result["level"]
        prediction_recommendation = result["recommendation"]
        prediction_message = result["message"]

    # YESTERDAY VS TODAY COMPARISON
    if selected_meter_id:

        comparison = get_daily_comparison(selected_meter_id)

        comparison_today = comparison["today"]
        comparison_yesterday = comparison["yesterday"]
        comparison_difference = comparison["difference"]
        comparison_status = comparison["status"]
        comparison_message = comparison["message"]
        comparison_recommendation = comparison["recommendation"]

    # GRAPH
    graph_labels, graph_values = get_consumption_graph_data()

    return render_template(
        "index.html",

        prediction=prediction,
        usage_level=usage_level,
        time_period=time_period,
        recommendation=recommendation,
        recommendation_title=recommendation_title,
        night_recommendation=night_recommendation,

        next_day_prediction=next_day_prediction,
        prediction_level=prediction_level,
        prediction_recommendation=prediction_recommendation,
        prediction_message=prediction_message,
        selected_meter_id=selected_meter_id,

        comparison_today=comparison_today,
        comparison_yesterday=comparison_yesterday,
        comparison_difference=comparison_difference,
        comparison_status=comparison_status,
        comparison_message=comparison_message,
        comparison_recommendation=comparison_recommendation,

        tracking_result=tracking_result,
        tracker_error=tracker_error,
        tracker_history=tracker_history,

        graph_labels=graph_labels,
        graph_values=graph_values
    )


# ---------------- ENERGY INSIGHTS ----------------

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
    return render_template("thankyou.html")


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    app.run(debug=True)
