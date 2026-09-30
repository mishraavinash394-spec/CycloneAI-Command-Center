# 🌪 CycloneAI - Intelligent Tropical Cyclone Monitoring & Prediction Command Center

> **Smart India Hackathon (SIH 26070) Project**  
> Developed for **MoES (Ministry of Earth Sciences)** & **IMD (India Meteorological Department)**.

---

## 📖 Table of Contents
1. [About the Project](#-about-the-project)
2. [Problem Statement & Motivation](#-problem-statement--motivation)
3. [Key Features & Capabilities](#-key-features--capabilities)
4. [Tech Stack](#-tech-stack)
5. [System Architecture & Workflow](#-system-architecture--workflow)
6. [Project Structure](#-project-structure)
7. [Installation & Setup Guide](#-installation--setup-guide)
8. [Usage & Navigation](#-usage--navigation)
9. [Future Enhancements](#-future-enhancements)

---

## 🚀 About the Project
**CycloneAI** is an advanced, end-to-end web-based command center platform engineered for real-time tropical cyclone monitoring, track prediction, and automated severity assessment. Designed specifically to support disaster management authorities, meteorologists, and response teams (aligned with MoES and IMD frameworks), the system bridges the gap between raw meteorological telemetry and actionable emergency intelligence.

---

## 🎯 Problem Statement & Motivation
Severe tropical cyclones cause massive loss of life and infrastructure damage in coastal regions due to delayed warnings or inaccurate track forecasting. Traditional monitoring systems often lack rapid, automated predictive analytics combined with interactive spatial visualization. **CycloneAI** solves this by providing:
- Real-time telemetry processing and data analysis.
- Automated machine learning-driven severity and track forecasting.
- Interactive GIS map visualization for evacuation planning.

---

## 🌟 Key Features & Capabilities

- **📊 Interactive Streamlit Control Center:** A highly responsive, clean dashboard providing live metrics, basin overviews, and predictive analytics controls.
- **🗺 Dynamic GIS Mapping:** Powered by **Folium** and **OpenStreetMap**, visualizing cyclone paths, wind-speed buffer radii, and landfall prediction cones.
- **🤖 Machine Learning Prediction Pipeline:** Leverages pre-trained regression and classification models serialized using **Pickle** and built on **Scikit-learn** to forecast storm intensity, wind speeds, and trajectory coordinates.
- **📈 Advanced Data Processing:** High-speed parsing, cleaning, and filtering of large meteorological datasets using **Pandas** and **NumPy**.
- **🚨 Automated Alert & Risk Assessment:** Automatically categorizes storm severity into Red, Orange, and Green alert levels based on real-time wind speed thresholds.

---

## 🛠 Tech Stack

Our application is built using a robust, Python-centric architecture optimized for data science, machine learning, and rapid geospatial web deployment:

* **Frontend:** Python, Streamlit, HTML/CSS
* **Backend & Data Processing:** Python, Pandas, NumPy
* **GIS / Mapping:** Folium, OpenStreetMap
* **AI / ML:** Scikit-learn, Pickle

---

## ⚙️ System Architecture & Workflow

1. **Data Ingestion:** Historical tracking datasets and live meteorological feeds are processed via **Pandas** and **NumPy**.
2. **AI / ML Inference:** User inputs or live telemetry data are passed through the **Scikit-learn** model (loaded via **Pickle**) to predict storm behavior and intensity.
3. **Geospatial Rendering:** Predicted coordinates and tracks are rendered dynamically onto interactive maps using **Folium** and **OpenStreetMap**.
4. **Actionable UI Presentation:** Results, confidence scores, and alert statuses are instantly displayed on the **Streamlit** dashboard.

---

## 📂 Project Structure

```text
CycloneAI-Command-Center/
├── app.py                 # Main Streamlit application entry point
├── model.pkl              # Pre-trained Scikit-learn machine learning model
├── data/                  # Historical cyclone datasets and telemetry logs
│   └── ibtracs_lite.csv   # Cyclone tracking database sample
├── requirements.txt       # Project dependencies list
└── README.md              # Detailed project documentation
