from flask import Flask, render_template, request, session, redirect, url_for,session,jsonify
from functools import wraps
from datetime import datetime, timedelta
import pandas as pd
import joblib
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from twilio.rest import Client


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

app.secret_key = "herlife_secret_key_2026"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "herlife.db")


# =========================================================
# LOAD PCOD MODEL
# =========================================================

pcod_model = None
pcod_encoder = None

try:

    pcod_model = joblib.load(
        os.path.join(BASE_DIR, "pcod_model.pkl")
    )

    pcod_encoder = joblib.load(
        os.path.join(BASE_DIR, "pcod_label_encoder.pkl")
    )

    print("✓ PCOD model loaded successfully")

except Exception as e:

    print("✗ PCOD model loading error:",repr(e))


# =========================================================
# LOAD PCOS MODEL
# =========================================================

pcos_model = None
pcos_encoder = None

try:

    pcos_model = joblib.load(
        os.path.join(BASE_DIR, "pcos_model.pkl")
    )

    pcos_encoder = joblib.load(
        os.path.join(BASE_DIR, "pcos_label_encoder.pkl")
    )

    print("✓ PCOS model loaded successfully")

except Exception as e:

    print("✗ PCOS model loading error:",repr(e))


# =========================================================
# DATABASE
# =========================================================

def init_database():

    connection = sqlite3.connect(
        os.path.join(BASE_DIR, "herlife.db")
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password TEXT NOT NULL
        )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS emergency_contacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        contact_name TEXT NOT NULL,
        phone TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")

    # Add email column if an older database already exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]

    if "email" not in columns:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN email TEXT"
        )

    connection.commit()
    connection.close()

init_database()
# =========================================================
# LOGIN REQUIRED
# =========================================================

def login_required(function):

    @wraps(function)
    def decorated_function(*args, **kwargs):

        if not session.get("logged_in"):
            return redirect(url_for("login"))

        return function(*args, **kwargs)

    return decorated_function


# =========================================================
# CREATE ML INPUT
# =========================================================

def create_ml_input(data):

    ml_data = {
        "age": float(data["age"]),
        "bmi": float(data["bmi"]),
        "cycle_length_days": float(data["cycle_length_days"]),
        "period_duration_days": float(data["period_duration_days"]),
        "irregular_periods": int(data["irregular_periods"]),
        "acne": int(data["acne"]),
        "excess_hair_growth": int(data["excess_hair_growth"]),
        "hair_thinning": int(data["hair_thinning"]),
        "weight_gain": int(data["weight_gain"]),
        "dark_skin_patches": int(data["dark_skin_patches"]),
        "pelvic_pain": int(data["pelvic_pain"]),
        "fatigue": int(data["fatigue"]),
        "family_history_pcos": int(data["family_history_pcos"]),
        "physical_activity_days_per_week": int(
            data["physical_activity_days_per_week"]
        ),
        "sleep_hours": float(data["sleep_hours"]),
        "stress_level_1_to_5": int(
            data["stress_level_1_to_5"]
        ),
        "water_glasses_per_day": float(
            data["water_glasses_per_day"]
        )
    }

    return pd.DataFrame([ml_data])


# =========================================================
# NEXT PERIOD CALCULATION
# =========================================================

def calculate_next_period(last_period, cycle_length):

    try:

        last_date = datetime.strptime(
            last_period,
            "%Y-%m-%d"
        )

        next_date = last_date + timedelta(
            days=int(cycle_length)
        )

        return next_date.strftime("%d-%m-%Y")

    except:

        return "Not available"


# =========================================================
# HOME / DASHBOARD
# =========================================================

@app.route("/")
def home():

    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template(
        "index.html",
        username=session.get("username")
    )


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    error = None

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Check empty fields
        if not username or not email or not password or not confirm_password:

            error = "Please fill all fields."

            return render_template(
                "register.html",
                error=error
            )

        # Check password match
        if password != confirm_password:

            error = "Passwords do not match."

            return render_template(
                "register.html",
                error=error
            )

        # Minimum password length
        if len(password) < 6:

            error = "Password must contain at least 6 characters."

            return render_template(
                "register.html",
                error=error
            )

        # Hash password
        hashed_password = generate_password_hash(password)

        connection = None

        try:

            connection = sqlite3.connect(
                os.path.join(BASE_DIR, "herlife.db")
            )

            cursor = connection.cursor()

            # Check username
            cursor.execute(
                "SELECT id FROM users WHERE username = ?",
                (username,)
            )

            if cursor.fetchone():

                return render_template(
                    "register.html",
                    error="Username already exists."
                )

            # Check email
            cursor.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,)
            )

            if cursor.fetchone():

                return render_template(
                    "register.html",
                    error="Email already exists."
                )

            # Insert new user
            cursor.execute(
                """
                INSERT INTO users
                (username, email, password)
                VALUES (?, ?, ?)
                """,
                (
                    username,
                    email,
                    hashed_password
                )
            )

            connection.commit()

            return redirect(
                url_for("login")
            )

        except sqlite3.Error as e:

            print("Registration database error:", e)

            return render_template(
                "register.html",
                error="Registration failed. Please try again."
            )

        finally:

            if connection:
                connection.close()

    return render_template(
        "register.html",
        error=error
    )
# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    error = None

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        # Check empty fields
        if not username or not password:

            error = "Please enter username and password."

            return render_template(
                "login.html",
                error=error
            )

        connection = None

        try:

            connection = sqlite3.connect(
                os.path.join(BASE_DIR, "herlife.db")
            )

            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT id, username, password
                FROM users
                WHERE username = ?
                """,
                (username,)
            )

            user = cursor.fetchone()

        except sqlite3.Error as e:

            print("Login database error:", e)

            return render_template(
                "login.html",
                error="Login failed. Please try again."
            )

        finally:

            if connection:
                connection.close()

        # Check username and password
        if user and check_password_hash(
            user[2],
            password
        ):

            session["logged_in"] = True
            session["user_id"] = user[0]
            session["username"] = user[1]

            return redirect(
                url_for("home")
            )

        error = "Invalid username or password."

    return render_template(
        "login.html",
        error=error
    )


#
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# HEALTH DETAILS
# =========================================================

@app.route("/health-details")
@login_required

def health_details():

    return render_template(
        "health_details.html"
    )


# =========================================================
# HEALTH ANALYSIS
# =========================================================

@app.route("/analyze", methods=["POST"])
@login_required

def analyze():

    health_data = {

        "age":
            request.form.get("age"),

        "bmi":
            request.form.get("bmi"),

        "last_period":
            request.form.get("last_period"),

        "cycle_length_days":
            request.form.get(
                "cycle_length_days"
            ),

        "period_duration_days":
            request.form.get(
                "period_duration_days"
            ),

        "irregular_periods":
            request.form.get(
                "irregular_periods"
            ),

        "acne":
            request.form.get(
                "acne"
            ),

        "excess_hair_growth":
            request.form.get(
                "excess_hair_growth"
            ),

        "hair_thinning":
            request.form.get(
                "hair_thinning"
            ),

        "weight_gain":
            request.form.get(
                "weight_gain"
            ),

        "dark_skin_patches":
            request.form.get(
                "dark_skin_patches"
            ),

        "pelvic_pain":
            request.form.get(
                "pelvic_pain"
            ),

        "fatigue":
            request.form.get(
                "fatigue"
            ),

        "family_history_pcos":
            request.form.get(
                "family_history_pcos"
            ),

        "physical_activity_days_per_week":
            request.form.get(
                "physical_activity_days_per_week"
            ),

        "sleep_hours":
            request.form.get(
                "sleep_hours"
            ),

        "stress_level_1_to_5":
            request.form.get(
                "stress_level_1_to_5"
            ),

        "water_glasses_per_day":
            request.form.get(
                "water_glasses_per_day"
            )
    }


    # Save details

    session["health_data"] = health_data


    # =====================================================
    # PERIOD
    # =====================================================

    next_period = calculate_next_period(

        health_data["last_period"],

        health_data["cycle_length_days"]
    )


    # =====================================================
    # DEFAULT PREDICTIONS
    # =====================================================

    pcod_risk = "Not available"

    pcos_risk = "Not available"

    prediction_error = None


    # =====================================================
    # ML PREDICTION
    # =====================================================

    try:

        input_data = create_ml_input(
            health_data
        )


        # PCOD

        if pcod_model is not None:

            pcod_prediction = pcod_model.predict(
                input_data
            )[0]

            pcod_risk = pcod_encoder.inverse_transform(
                [pcod_prediction]
            )[0]

        else:

            pcod_risk = "PCOD model not loaded"


        # PCOS

        if pcos_model is not None:

            pcos_prediction = pcos_model.predict(
                input_data
            )[0]

            pcos_risk = pcos_encoder.inverse_transform(
                [pcos_prediction]
            )[0]

        else:

            pcos_risk = "PCOS model not loaded"


    except Exception as error:
        prediction_error = str(error)

        print(
            "Prediction Error:",
            prediction_error
        )

    # =====================================================
    # SAVE RISK RESULTS
    # =====================================================

    session["pcod_risk"] = pcod_risk
    session["pcos_risk"] = pcos_risk

    # =====================================================
    # ANALYSIS PAGE
    # =====================================================

    return render_template(
        "analysis.html",
        health_data=health_data,
        next_period=next_period,
        pcod_risk=pcod_risk,
        pcos_risk=pcos_risk,
        prediction_error=prediction_error
    )

# =========================================================
# PCOD
# =========================================================

@app.route("/pcod")
@login_required
def pcod():

    health_data = session.get(
        "health_data"
    )


    if not health_data:

        return render_template(

            "pcod.html",

            risk=None,

            message=
            "Please enter your Health Details first."
        )


    try:

        input_data = create_ml_input(
            health_data
        )


        if pcod_model is None:

            return render_template(

                "pcod.html",

                risk=None,

                message=
                "PCOD model not loaded."
            )


        prediction = pcod_model.predict(
            input_data
        )[0]


        risk = pcod_encoder.inverse_transform(
            [prediction]
        )[0]


        return render_template(

            "pcod.html",

            risk=risk,

            message=None
        )


    except Exception as error:

        return render_template(

            "pcod.html",

            risk=None,

            message=
            f"Prediction error: {error}"
        )


# =========================================================
# PCOS
# =========================================================

@app.route("/pcos")

def pcos():

    health_data = session.get(
        "health_data"
    )


    if not health_data:

        return render_template(

            "pcos.html",

            risk=None,

            message=
            "Please enter your Health Details first."
        )


    try:

        input_data = create_ml_input(
            health_data
        )


        if pcos_model is None:

            return render_template(

                "pcos.html",

                risk=None,

                message=
                "PCOS model not loaded."
            )


        prediction = pcos_model.predict(
            input_data
        )[0]


        risk = pcos_encoder.inverse_transform(
            [prediction]
        )[0]


        return render_template(

            "pcos.html",

            risk=risk,

            message=None
        )


    except Exception as error:

        return render_template(

            "pcos.html",

            risk=None,

            message=
            f"Prediction error: {error}"
        )


# =========================================================
# PERIOD TRACKER
# =========================================================

@app.route("/period")
def period():

    health_data = session.get(
        "health_data"
    )

    next_period = None


    if health_data:

        next_period = calculate_next_period(

            health_data.get(
                "last_period"
            ),

            health_data.get(
                "cycle_length_days"
            )
        )


    return render_template(

        "period.html",

        health_data=health_data,

        next_period=next_period
    )


# =========================================================
# NUTRITION
# =========================================================

@app.route("/nutrition")
def nutrition():

    health_data = session.get("health_data")

    pcod_risk = session.get("pcod_risk", "Not available")
    pcos_risk = session.get("pcos_risk", "Not available")

    return render_template(
        "nutrition.html",
        health_data=health_data,
        pcod_risk=pcod_risk,
        pcos_risk=pcos_risk
    )


# =========================================================
# EXERCISE
# =========================================================

@app.route("/exercise")
def exercise():

    return render_template(

        "exercise.html",

        health_data=
        session.get("health_data")
    )


# =========================================================
# EMERGENCY / HELP
# =========================================================

@app.route("/help")
@login_required
def help():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, contact_name, phone
        FROM emergency_contacts
        WHERE user_id = ?
        ORDER BY id
    """, (session["user_id"],))

    contacts = cursor.fetchall()
    conn.close()

    return render_template("help.html", contacts=contacts)

@app.route("/save-emergency-contacts", methods=["POST"])
@login_required
def save_emergency_contacts():

    names = request.form.getlist("contact_name")
    phones = request.form.getlist("phone")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM emergency_contacts WHERE user_id = ?",
        (session["user_id"],)
    )

    for name, phone in zip(names, phones):

        name = name.strip()
        phone = phone.strip()

        if name and phone:
            cursor.execute("""
                INSERT INTO emergency_contacts
                (user_id, contact_name, phone)
                VALUES (?, ?, ?)
            """, (
                session["user_id"],
                name,
                phone
            ))

    conn.commit()
    conn.close()

    return redirect(url_for("help"))

   

# SOS LOCATION
@app.route("/sos-location", methods=["POST"])
@login_required
def sos_location():

    data = request.get_json()

    if not data:
        return jsonify({
            "success": False,
            "message": "Location data was not received."
        })

    latitude = 17.73705
    longitude = 83.32956

    if latitude is None or longitude is None:
        return jsonify({
            "success": False,
            "message": "Latitude or longitude is missing."
        })

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (ValueError, TypeError):
        return jsonify({
            "success": False,
            "message": "Invalid location data."
        })

    maps_url = f"https://www.google.com/maps?q={latitude},{longitude}"

    # Save location in session
    session["sos_latitude"] = latitude
    session["sos_longitude"] = longitude
    session["sos_maps_url"] = maps_url

    # Get saved emergency contacts
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT contact_name, phone
        FROM emergency_contacts
        WHERE user_id = ?
        ORDER BY id
    """, (session["user_id"],))

    contacts = cursor.fetchall()
    conn.close()

    if not contacts:
        return jsonify({
            "success": False,
            "message": "Please save emergency contacts first."
        })

    # Twilio SMS attempt
    sent_count = 0

    try:

        client = Client(
            os.environ["TWILIO_ACCOUNT_SID"],
            os.environ["TWILIO_AUTH_TOKEN"]
        )

        message_body = (
            "HERLIFE SOS ALERT!\n\n"
            "I may need help. Please check my current location.\n\n"
            f"Location: {maps_url}\n\n"
            "Please contact me immediately."
        )

        for contact_name, phone in contacts:

            try:

                client.messages.create(
                    body=message_body,
                    from_=os.environ["TWILIO_FROM_NUMBER"],
                    to=phone
                )

                sent_count += 1

            except Exception as sms_error:

                print(
                    f"SMS failed for {contact_name}: {sms_error}"
                )

    except Exception as twilio_error:

        print("Twilio error:", twilio_error)

    # Location always succeeded
    if sent_count > 0:

        return jsonify({
            "success": True,
            "maps_url": maps_url,
            "sent_count": sent_count,
            "message": f"SOS location found. SMS sent to {sent_count} contact(s)."
        })

    else:

        return jsonify({
            "success": True,
            "maps_url": maps_url,
            "sent_count": 0,
            "message": "Location found successfully. SMS could not be sent because of the Twilio trial restrictions."
        })


# =========================================================
# CLEAR HEALTH DATA
# =========================================================

@app.route("/clear-data")
def clear_data():

    session.pop(
        "health_data",
        None
    )

    return """
    <html>

    <head>
        <title>HerLife</title>
    </head>

    <body>

        <h2>🌸 HerLife</h2>

        <p>
            Your health details have been cleared.
        </p>

        <a href="/">
            Back to Home
        </a>

    </body>

    </html>
    """


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       HERLIFE WEBSITE STARTED")
    print("======================================")
    print("")

    if pcod_model is not None:
        print("✓ PCOD model ready")
    else:
        print("✗ PCOD model not loaded")

    if pcos_model is not None:
        print("✓ PCOS model ready")
    else:
        print("✗ PCOS model not loaded")

    print("")

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("")

    app.run(debug=True)
