import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysqldb import MySQL
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

app = Flask(__name__)
app.secret_key = "satellite_super_secret_key"

# --- Database Config ---
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '' 
app.config['MYSQL_DB'] = 'satellite_db'
mysql = MySQL(app)

# --- Upload Folder ---
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

# --- AI Model Setup ---
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(45, activation='softmax')
])
model.build((None, 224, 224, 3))
model.load_weights("mobilenet_weights.weights.h5")

class_names = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland']

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('home.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cur.fetchone()
        cur.close()
        if user:
            session['username'] = username
            return redirect(url_for('dashboard'))
        flash("Invalid Credentials!")
    return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # 1. Grab data from the HTML form names
        username = request.form['username']
        password = request.form['password']
        
        # 2. Connect to MySQL
        cur = mysql.connection.cursor()
        
        try:
            # 3. Insert the new user into the 'users' table
            cur.execute("INSERT INTO users(username, password) VALUES (%s, %s)", (username, password))
            
            # 4. Save changes
            mysql.connection.commit()
            cur.close()
            
            # 5. Send them to the login page after success
            return redirect(url_for('login'))
        except Exception as e:
            return f"Registration Error: {e}"
            
    # If they just visit the page, show them the HTML
    return render_template('register.html')

@app.route("/dashboard")
def dashboard():
    if 'username' not in session: return redirect(url_for('login'))
    return render_template("dashboard.html")

@app.route('/predict', methods=['POST'])
def predict():
    # --- SECURITY LOCK ---
    # If 'username' is not in the session, redirect to login
    if 'username' not in session:
        flash("Unauthorized access! Please login to use the AI scanner.")
        return redirect(url_for('login'))
    # ---------------------

    file = request.files.get("file")
    if not file or file.filename == "":
        return "No file selected"

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # AI Prediction Logic
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(np.max(prediction) * 100, 2)

    return render_template(
        "results.html",
        prediction=predicted_class,
        confidence=confidence,
        image_name=file.filename # Ensure this matches your results.html variable
    )
@app.route('/home')
def home_redirect():
    # This sends users to the dashboard or a welcome page
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # We pull user info from the session
    user_data = {
        'name': session['username'],
        'role': 'Satellite Operator',
        'joined': 'May 2026'
    }
    return render_template('profile.html', user=user_data)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)