import os
import pickle
import pandas as pd
import sqlite3
from flask import Flask, request, jsonify, render_template, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, template_folder='.')
app.secret_key = os.environ.get('SECRET_KEY', 'default-apex-secret-key-12345')

DATABASE = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        with get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Database initialization error: {e}")

# Run database initialization
init_db()


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response

# Global variables for model and preprocessing artifacts
model = None
model_name = None
encoders = None
scaler = None
feature_cols = None
numeric_cols = None
categorical_cols = None

def load_model():
    """
    Loads the trained model and preprocessing steps from pickle.
    """
    global model, model_name, encoders, scaler, feature_cols, numeric_cols, categorical_cols
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Model file not found at: {model_path}. Running train_model.py automatically...")
        try:
            import train_model
            train_model.main()
        except Exception as e:
            print(f"Failed to auto-train model: {str(e)}")
            return False
        
    try:
        with open(model_path, 'rb') as f:
            payload = pickle.load(f)
            
        model = payload['model']
        model_name = payload['model_name']
        encoders = payload['encoders']
        scaler = payload['scaler']
        feature_cols = payload['feature_cols']
        numeric_cols = payload['numeric_cols']
        categorical_cols = payload['categorical_cols']
        
        print(f"Successfully loaded model '{model_name}' (Accuracy: {payload.get('accuracy', 0.0):.4f})")
        return True
    except Exception as e:
        print(f"Error loading model payload: {str(e)}")
        return False

# Initialize the model on startup
model_loaded = load_model()

@app.route('/')
def home():
    """
    Renders the homepage dashboard.
    """
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'status': 'error', 'message': 'Username and password are required.'}), 400
    
    username = data['username'].strip()
    password = data['password']
    
    if len(username) < 3:
        return jsonify({'status': 'error', 'message': 'Username must be at least 3 characters long.'}), 400
    if len(password) < 6:
        return jsonify({'status': 'error', 'message': 'Password must be at least 6 characters long.'}), 400
        
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            if cursor.fetchone():
                return jsonify({'status': 'error', 'message': 'Username already exists.'}), 400
            
            hashed_pw = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed_pw))
            conn.commit()
            
        return jsonify({'status': 'success', 'message': 'Registration successful.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Registration failed: {str(e)}'}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'status': 'error', 'message': 'Username and password are required.'}), 400
        
    username = data['username'].strip()
    password = data['password']
    
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password_hash'], password):
                session['username'] = user['username']
                return jsonify({'status': 'success', 'message': 'Login successful.', 'username': user['username']})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid username or password.'}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Login failed: {str(e)}'}), 500

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'status': 'success', 'message': 'Logged out successfully.'})

@app.route('/check-session', methods=['GET'])
def check_session():
    if 'username' in session:
        return jsonify({
            'status': 'success',
            'authenticated': True,
            'username': session['username']
        })
    return jsonify({
        'status': 'success',
        'authenticated': False
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
        
    # Check if user is logged in
    if 'username' not in session:
        return jsonify({
            'status': 'error',
            'message': 'Unauthorized. Please log in first.'
        }), 401

    """
    Handles JSON prediction requests.
    Expects keys: Gender, Income Type, Annual Income, Employment Duration, Education Level
    """
    global model, encoders, scaler, model_loaded
    
    # Reload model if it wasn't loaded successfully at startup (e.g. if training finished after app started)
    if not model_loaded:
        model_loaded = load_model()
        if not model_loaded:
            return jsonify({
                'status': 'error',
                'message': 'Model is not trained or loaded. Please run the training script first.'
            }), 500

    # Parse and validate request JSON
    data = request.get_json()
    if not data:
        return jsonify({
            'status': 'error',
            'message': 'No input data provided. Request must contain a valid JSON payload.'
        }), 400

    # Required fields validation
    required_fields = ['Gender', 'Income Type', 'Annual Income', 'Employment Duration', 'Education Level']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({
            'status': 'error',
            'message': f"Missing required fields: {', '.join(missing_fields)}"
        }), 400

    try:
        # Validate values and data types
        try:
            annual_income = float(data['Annual Income'])
            if annual_income < 0:
                raise ValueError("Annual Income cannot be negative.")
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': "Annual Income must be a positive number."
            }), 400

        try:
            employment_duration = float(data['Employment Duration'])
            if employment_duration < 0:
                raise ValueError("Employment Duration cannot be negative.")
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': "Employment Duration must be a positive number (years)."
            }), 400

        # Validate categorical values against encoder classes
        encoded_values = {}
        for col in categorical_cols:
            val = data[col]
            le = encoders[col]
            if val not in le.classes_:
                return jsonify({
                    'status': 'error',
                    'message': f"Invalid value '{val}' for field '{col}'. Must be one of {list(le.classes_)}."
                }), 400
            
            # Encode categorical values
            encoded_values[col] = le.transform([val])[0]

        # Standardize numeric values
        # Scaler expects a 2D array matching the shapes of original numerical features (Annual Income, Employment Duration)
        numeric_df = pd.DataFrame([[annual_income, employment_duration]], columns=numeric_cols)
        scaled_numeric = scaler.transform(numeric_df)
        
        # Assemble feature vector in correct order
        # Ordered features: ['Gender', 'Income Type', 'Annual Income', 'Employment Duration', 'Education Level']
        row_df = pd.DataFrame([{
            'Gender': encoded_values['Gender'],
            'Income Type': encoded_values['Income Type'],
            'Annual Income': scaled_numeric[0][0],
            'Employment Duration': scaled_numeric[0][1],
            'Education Level': encoded_values['Education Level']
        }])
        
        # Rearrange columns to match exact feature column order of training
        row_df = row_df[feature_cols]

        # Predict outcome (0 or 1)
        prediction = int(model.predict(row_df)[0])
        
        # Calculate prediction probability/confidence
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(row_df)[0][1])
        else:
            # If the model does not support predict_proba, use decision function or hard status
            prob = 1.0 if prediction == 1 else 0.0

        return jsonify({
            'status': 'success',
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'probability': prob,
            'model_used': model_name
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Prediction failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Use debug=True for local development, run on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
