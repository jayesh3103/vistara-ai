# 🇮🇳 Vistara-AI: Predictive Demographic Analytics for UIDAI

**Vistara-AI** is a cutting-edge analytics dashboard designed to empower UIDAI administrators with predictive insights into Aadhaar enrolment and update trends. By leveraging Machine Learning and Geospatial Analytics, it transforms raw data into actionable intelligence for resource optimization and fraud detection.

![Vistara-AI Banner](https://img.shields.io/badge/Status-Hackathon%20Ready-green) ![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)

## 🌟 Key Features

### 1. 🗺️ The Pulse Map (Geospatial Intelligence)

- **Live Visualization**: Interactive map showing "Update Velocity" across all Indian states.
- **Risk Overlay**: Color-coded stress levels to instantly identify high-activity zones.
- **Tech**: Powered by Plotly and Mapbox (carto-positron).

### 2. 🚨 Anomaly Hunter (Machine Learning)

- **Automated Detection**: Uses **Isolation Forest (Unsupervised Learning)** to flag districts with irregular update patterns.
- **Metrics**: Analyzes Velocity, Divergence Ratio (Bio vs. Demo updates), and Migration Index.
- **Drill-Down**: Detailed "Risk Cards" for flagged districts.

### 3. 🔮 Decision Support System (The "Game Changer")

- **"What-If" Simulator**: Interactive slider to simulate resource interventions (e.g., "Deploying 5 Mobile Vans") and visualize the impact on future trends.
- **Forecasting**: 3-Month Projection of update volume using Time-Series analysis.
- **Automated Policy Briefs**: Generates download-ready text summaries with AI-driven recommendations for bureaucrats.

### 4. 📈 Strategic Insights

- **Growth vs. Churn**: Quadrant analysis comparing New Enrolments vs. Demographic Updates.
- **Migration Detection**: Identifies "Phantom Growth" where updates significantly outpace new enrolments.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/jayesh3103/vistara-ai.git
    cd vistara-ai
    ```

2.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure

```
vistara-ai/
├── 📄 app.py              # Main Dashboard Application (UI/UX)
├── 📄 analytics.py        # ML Models (Isolation Forest) & Forecasting Logic
├── 📄 data_processor.py   # Data Cleaning, Standardization & Pipeline
├── 📂 datasets/           # Source CSV Files (Enrolment, Update, Biometric)
├── 📄 requirements.txt    # Python Dependencies
└── 📄 README.md           # Project Documentation
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Isolation Forest)
- **Visualization**: Plotly Express, Mapbox

---

## 📢 Hackathon Context

This project addresses the **"Predicting Demographic Changes & Service Demand"** problem statement. It moves beyond static reporting to provide **predictive** and **prescriptive** analytics, enabling proactive governance.
