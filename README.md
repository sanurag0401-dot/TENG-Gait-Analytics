# TENG-Gait Analytics
### Experimental Signal Processing & Intelligent Gait Analysis Platform

TENG-Gait Analytics is a Python-based scientific data analysis platform developed to explore gait biomechanics, signal processing and intelligent analysis of sensor-generated data.

The project combines force-plate-based gait analysis using publicly available experimental data with a separate workflow for analysing user-uploaded triboelectric nanogenerator (TENG) voltage signals.

The platform aims to provide an accessible environment for processing experimental signals, extracting meaningful features, visualising results and exploring patterns in gait-related measurements.

## Project Objectives

- Process and visualise biomechanical signals.
- Apply digital filtering techniques to reduce signal noise.
- Detect approximate gait contact events from force-plate measurements.
- Extract relevant temporal and statistical gait features.
- Perform exploratory statistical analysis and machine learning.
- Provide a separate interface for analysing experimental TENG voltage data.
- Present analysis results through an interactive Streamlit dashboard.

## Key Features

### 1. Force-Plate Gait Analysis
- Import and inspect force-plate datasets.
- Visualise raw vertical ground reaction force signals.
- Apply Butterworth low-pass filtering.
- Detect approximate contact intervals using threshold-based detection.
- Extract stance duration, peak force, mean force, standard deviation and force impulse.

### 2. Statistical Analysis
- Descriptive statistical summaries.
- Interquartile range (IQR)-based outlier identification.
- Pearson correlation analysis.
- Correlation matrix visualisation.

### 3. Machine Learning
- Feature standardisation using StandardScaler.
- K-Means clustering.
- Silhouette score comparison.
- Principal Component Analysis (PCA) for dimensionality reduction and visualisation.

### 4. TENG Voltage Analysis
A separate custom-data workflow is provided for uploading experimental TENG voltage measurements.

The workflow supports:
- CSV data upload.
- Measurement-column selection.
- Voltage unit selection.
- User-defined sampling frequency.
- Optional signal filtering.
- Voltage signal visualisation.
- Basic electrical signal statistics.

Additional experimental signal analysis will be developed as validated TENG datasets become available.

## Technology Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data processing | NumPy, Pandas |
| Signal processing | SciPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn |
| Dashboard | Streamlit |
| Development | Visual Studio Code |

## Project Structure

```text
gait-python/
│
├── data/
│   ├── raw/
│   │   ├── physionet/
│   │   └── triboelectric/
│   └── processed/
│       └── physionet/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── features/
│   ├── gait_analysis/
│   │
│   ├── inspect_dataset.py
│   ├── plot_force.py
│   ├── filter_signal.py
│   ├── gait_event_detection.py
│   ├── feature_extraction.py
│   ├── feature_visualization.py
│   ├── statistical_analysis.py
│   └── clustering_analysis.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The force-plate gait-analysis workflow uses the **PhysioNet Multimodal Gait Dataset**.

Dataset source: https://physionet.org/content/multimodal-gait-dataset/1.0.0/

The dataset contains multimodal recordings collected from healthy participants, including force-plate and other physiological measurements.

The current development workflow uses a force-plate recording to demonstrate gait signal processing and feature extraction.

**Important:** Force-plate measurements and TENG voltage measurements are treated as separate data sources. The public force-plate dataset is not presented as experimental TENG output.

Please obtain the dataset from its original source and follow its licensing and attribution requirements.

## Methodology

The force-plate analysis follows this general pipeline:

1. Dataset inspection and preprocessing.
2. Vertical force signal extraction.
3. Butterworth low-pass filtering.
4. Threshold-based contact interval detection.
5. Gait feature extraction.
6. Descriptive and correlation analysis.
7. K-Means clustering and PCA visualisation.

The TENG workflow is maintained separately and processes uploaded voltage measurements using their selected measurement column, voltage unit and sampling frequency.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd gait-python
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
python -m streamlit run src/dashboard/app.py
```

The Streamlit application will open in your browser.

## Scientific Considerations and Limitations

- Contact intervals are estimated using a force threshold and should not automatically be interpreted as independently validated heel-strike and toe-off events.
- The current force-plate analysis is based on a limited participant recording and should be considered exploratory.
- K-Means clustering is exploratory and does not establish clinically meaningful gait categories.
- Signal filtering parameters should be selected according to the measurement system and research objective.
- Reliable TENG power and energy calculations require additional electrical information, including load conditions and current measurements.
- Further validation with experimental TENG recordings and appropriately labelled gait trials is required.

## Future Development

- Advanced TENG voltage signal processing.
- Positive and negative peak detection.
- RMS and peak-to-peak voltage analysis.
- Frequency-domain analysis using FFT.
- Comparison of different TPU patterning configurations.
- Improved gait event detection and validation.
- Integration of additional experimental datasets.
- Enhanced interactive reporting and export functionality.

## Author

**Anurag Sehrawat**

M.Sc. Physics | Scientific Data Analysis | Signal Processing | Sensor Technology

This project reflects an interdisciplinary interest in physics, materials science, triboelectric nanogenerators, biomechanics and computational data analysis.

---

*Developed as an independent scientific and computational project.*