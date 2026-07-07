<<<<<<< HEAD
<<<<<<< HEAD
# Credit Card Approval Prediction Web Application

An end-to-end Machine Learning classification web application that predicts credit card approval decisions in real-time. The system evaluates candidate risk profiles based on Gender, Income Type, Annual Income, Employment Duration, and Education Level.

This project is fully modularized, containing separate scripts for the machine learning pipeline, a Flask backend API, and a modern glassmorphic responsive frontend dashboard.

---

## Features

1. **Machine Learning Pipeline**:
   - Automated synthetic dataset generation (`credit_card_data.csv`) simulating real-world credit risk profiles.
   - Robust preprocessing including **Label Encoding** for categorical features and **StandardScaler** for numeric features.
   - Comparative training and evaluation of 4 algorithms:
     - **Logistic Regression**
     - **Decision Tree Classifier**
     - **Random Forest Classifier**
     - **XGBoost Classifier**
   - Automatically selects the model with the highest test accuracy and exports it to `model.pkl`.

2. **Flask Backend API**:
   - Serves the frontend web dashboard.
   - Exposes a POST `/predict` API.
   - Validates user input types, boundaries, and categories.
   - Preprocesses client data and performs real-time model inference.

3. **Modern UI/UX Frontend**:
   - **Glassmorphic Design**: Sleek dark dashboard with backdrop-filter blurs and vibrant animated floating gradients.
   - **Responsive Layout**: Adapts gracefully to desktop, tablet, and mobile screens.
   - **Interactive Elements**: Hover animations, input glow states, and an active model status badge.
   - **Dynamic Outcome Display**:
     - **Approved (Emerald Green)** with an SVG circular progress ring indicating the model's approval probability.
     - **Rejected (Crimson Red)** with an SVG circular progress ring indicating the rejection confidence.
   - **Error Handling**: Displays validation warnings and connection errors dynamically in the console.

---

## File Structure

```
Credit Card Approval Prediction/
├── app.py                  # Flask server and prediction API
├── train_model.py          # ML data prep, training, and model selection
├── model.pkl               # Serialized best model, scaler, and encoders
├── credit_card_data.csv    # Generated synthetic dataset
├── requirements.txt        # Python dependency manifest
├── Dockerfile              # Docker container configuration
├── manifest.yml            # IBM Cloud Cloud Foundry deployment manifest
├── static/
│   ├── css/
│   │   └── style.css       # Core styling & glassmorphic aesthetics
│   └── js/
│       └── main.js         # AJAX fetch, form submission, and dynamic DOM updates
└── templates/
    └── index.html          # Dashboard HTML template structure
```

---

## Local Installation

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 1. Clone or Download the Project
Extract this workspace to your computer or navigate to the directory:
```bash
cd "Credit Card Approval Prediction"
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows (cmd/powershell):
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the ML Model
Execute the training pipeline to evaluate classifiers and output the `model.pkl` artifact:
```bash
python train_model.py
```
This generates the `credit_card_data.csv` dataset, trains the models, and prints the model comparisons.

### 5. Start the Web Server
Launch the Flask development server:
```bash
python app.py
```
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## IBM Cloud Deployment

You can deploy this application to IBM Cloud in two main ways: using **Cloud Foundry** or using **IBM Cloud Code Engine (Containers)**.

### Method A: Deploying via Cloud Foundry (Recommended)
IBM Cloud supports Cloud Foundry Python deployments out-of-the-box using the provided `manifest.yml`.

1. Install the [IBM Cloud CLI](https://cloud.ibm.com/docs/cli).
2. Log in to your IBM Cloud account:
   ```bash
   ibmcloud login
   ```
3. Target the Cloud Foundry API (and select your org/space):
   ```bash
   ibmcloud target --cf
   ```
4. Deploy the application:
   ```bash
   ibmcloud cf push
   ```
   *The Python Buildpack will automatically install requirements and the Flask server will startup using the start command specified in `manifest.yml`.*

### Method B: Deploying via IBM Cloud Code Engine (Containerized)
IBM Cloud Code Engine runs serverless containers from images.

1. Build the Docker image locally:
   ```bash
   docker build -t <your-registry>/credit-card-predictor:latest .
   ```
2. Push the image to IBM Cloud Container Registry (ICR) or Docker Hub:
   ```bash
   docker push <your-registry>/credit-card-predictor:latest
   ```
3. Deploy the container in the IBM Cloud Console under **Code Engine** -> **Projects** -> **Create Application**, pointing to the pushed container image. Specify port `8080` (which matches the Dockerfile).

---

## Git Operations

This repository is ready to be pushed to GitHub. Initialize git and push:
```bash
git init
git add .
git commit -m "Initial commit: Credit Card Approval Prediction System"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```
=======
# Credit-Card-Approval-Prediction
Credit Card Approval Prediction it shows the approval accept or reject
>>>>>>> 9aabc2705bb29155a423b0968428b47e57cf21c5
=======
# Credit-Card-Approval-Prediction-
>>>>>>> 710d72e8fe95089ec26a017b7941b64883d33eaf
