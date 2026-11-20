
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from scipy.signal import butter, sosfiltfilt


# =========================================
# 1. PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="TENG-Gait Analytics",
    page_icon="👣",
    layout="wide"
)

ROOT = Path(__file__).resolve().parents[2]

DATA_FOLDER = ROOT / "data" / "processed" / "physionet"

RAW_FILE = (
    ROOT / "data" / "raw" / "physionet"
    / "FP_S1_0.5.csv"
)

FEATURE_FILE = DATA_FOLDER / "gait_features_S1_0.5.csv"

EVENT_FILE = DATA_FOLDER / "FP_S1_0.5_gait_events.csv"

CLUSTER_FILE = DATA_FOLDER / "gait_clusters.csv"


# =========================================
# 2. LOAD DATA
# =========================================

@st.cache_data
def load_csv(file_path):

    if file_path.exists():
        return pd.read_csv(file_path)

    return None


features_df = load_csv(FEATURE_FILE)
events_df = load_csv(EVENT_FILE)
clusters_df = load_csv(CLUSTER_FILE)


# =========================================
# 3. SIDEBAR
# =========================================

st.sidebar.title("👣 TENG-Gait Analytics")

st.sidebar.caption(
    "Experimental Signal Processing & "
    "Intelligent Gait Analysis"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Gait Features",
        "Gait Events",
        "Machine Learning",
        "Raw Signal",
        "Custom Data Analysis"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Dataset: PhysioNet Multimodal Gait Dataset\n\n"
    "Participant: S1\n\n"
    "Walking speed: 0.5 m/s\n\n"
    "Signal: Force Plate 1 - Vertical Force"
)


# =========================================
# 4. MAIN HEADER
# =========================================

st.title("👣 TENG-Gait Analytics")

st.markdown(
    "### Intelligent Gait Analysis & Signal Processing Platform"
)

st.caption(
    "Exploratory analysis of force-plate gait measurements"
)

st.divider()


# =========================================
# 5. OVERVIEW PAGE
# =========================================

if page == "Overview":

    st.subheader("Project Overview")

    if features_df is None:

        st.error("Feature dataset not found.")

    else:

        analysis_df = features_df.copy()

        if "Recording Boundary Contact" in analysis_df.columns:

            analysis_df = analysis_df[
                analysis_df["Recording Boundary Contact"].astype(str).str.lower()
                == "false"
            ]

        total_contacts = len(analysis_df)

        avg_peak = analysis_df["Peak Force (N)"].mean()

        avg_stance = analysis_df["Stance Duration (s)"].mean()

        avg_mean_force = analysis_df["Mean Force (N)"].mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Detected Contacts",
            total_contacts
        )

        col2.metric(
            "Average Peak Force",
            f"{avg_peak:.2f} N"
        )

        col3.metric(
            "Average Stance Duration",
            f"{avg_stance:.3f} s"
        )

        col4.metric(
            "Average Mean Force",
            f"{avg_mean_force:.2f} N"
        )

        st.divider()

        st.subheader("Project Description")

        st.write(
            """
            TENG-Gait Analytics is a scientific data-analysis
            platform designed to explore gait-related signals,
            apply signal-processing techniques, extract
            biomechanical features and investigate patterns
            using machine learning.

            The current demonstration uses public force-plate
            measurements. TENG voltage data will be analysed
            through a separate pipeline when available.
            """
        )

        st.subheader("Feature Summary")

        st.dataframe(
            analysis_df.describe().round(3),
            use_container_width=True
        )


# =========================================
# 6. GAIT FEATURES PAGE
# =========================================

elif page == "Gait Features":

    st.subheader("Biomechanical Feature Analysis")

    if features_df is None:

        st.warning("Feature dataset is missing.")

    else:

        df = features_df.copy()

        if "Recording Boundary Contact" in df.columns:

            df = df[
                df["Recording Boundary Contact"].astype(str).str.lower()
                == "false"
            ]

        selected_feature = st.selectbox(
            "Select a feature",
            [
                "Peak Force (N)",
                "Mean Force (N)",
                "Stance Duration (s)",
                "Force Std (N)",
                "Force Impulse (N.s)"
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Mean",
            f"{df[selected_feature].mean():.3f}"
        )

        col2.metric(
            "Minimum",
            f"{df[selected_feature].min():.3f}"
        )

        col3.metric(
            "Maximum",
            f"{df[selected_feature].max():.3f}"
        )

        fig = px.histogram(
            df,
            x=selected_feature,
            nbins=12,
            title=f"Distribution of {selected_feature}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Feature Dataset")

        st.dataframe(
            df,
            use_container_width=True
        )


# =========================================
# 7. GAIT EVENTS PAGE
# =========================================

elif page == "Gait Events":

    st.subheader("Gait Contact Event Analysis")

    if events_df is None:

        st.warning("Gait event dataset is missing.")

    else:

        st.write(
            "Threshold-based contact intervals detected "
            "from the vertical force signal."
        )

        st.dataframe(
            events_df,
            use_container_width=True
        )

        if len(events_df) > 0:

            fig = px.scatter(
                events_df,
                x="Heel Strike (s)",
                y="Stance Duration (s)",
                title="Contact Duration Across Recording",
                labels={
                    "Heel Strike (s)": "Contact Start (seconds)"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================
# 8. MACHINE LEARNING PAGE
# =========================================

elif page == "Machine Learning":

    st.subheader("K-Means Clustering Analysis")

    if clusters_df is None:

        st.warning("Cluster dataset is missing.")

    else:

        st.write(
            "Exploratory grouping of detected contacts "
            "using standardised gait features."
        )

        st.metric(
            "Total Observations",
            len(clusters_df)
        )

        if "Cluster" in clusters_df.columns:

            distribution = (
                clusters_df["Cluster"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            distribution.columns = [
                "Cluster",
                "Observations"
            ]

            fig = px.bar(
                distribution,
                x="Cluster",
                y="Observations",
                title="Cluster Distribution",
                text="Observations"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Clustered Data")

        st.dataframe(
            clusters_df,
            use_container_width=True
        )


# =========================================
# 9. RAW SIGNAL PAGE
# =========================================

elif page == "Raw Signal":

    st.subheader("Force-Plate Signal Analysis")

    if not RAW_FILE.exists():

        st.warning("Raw force-plate dataset not found.")

    else:

        @st.cache_data
        def load_raw_signal(path):

            return pd.read_csv(path)

        raw_df = load_raw_signal(RAW_FILE)

        signal = raw_df["1:Fz"].to_numpy(dtype=float)

        fs = 1000

        time = np.arange(len(signal)) / fs

        cutoff = 20

        sos = butter(
            4,
            cutoff,
            btype="lowpass",
            fs=fs,
            output="sos"
        )

        filtered = sosfiltfilt(sos, signal)

        display_seconds = st.slider(
            "Select recording duration (seconds)",
            min_value=5,
            max_value=int(time[-1]),
            value=10
        )

        mask = time <= display_seconds

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time[mask],
                y=signal[mask],
                name="Raw Signal"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time[mask],
                y=filtered[mask],
                name="Filtered Signal"
            )
        )

        fig.update_layout(
            title="Raw vs Filtered Vertical Force",
            xaxis_title="Time (seconds)",
            yaxis_title="Vertical Force (N)",
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )



# =========================================
# 10. CUSTOM DATA ANALYSIS
# =========================================

elif page == "Custom Data Analysis":

    st.subheader("Custom Data Analysis")

    st.write(
        "Upload your own experimental CSV data and analyse "
        "your measurements independently of the built-in dataset."
    )

    st.info(
        "Your uploaded data is analysed separately. "
        "It does not replace or modify the PhysioNet results."
    )

    # -------------------------------------
    # 1. SELECT DATA TYPE
    # -------------------------------------

    analysis_type = st.radio(
        "Select your data type",
        [
            "Force-Plate Gait Data",
            "TENG Voltage Data"
        ],
        horizontal=True
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        key="custom_csv_upload"
    )

    if uploaded_file is not None:

        try:

            custom_df = pd.read_csv(uploaded_file)

            if custom_df.empty:
                st.error("The uploaded CSV file is empty.")

            else:
                st.success("CSV file uploaded successfully!")

                # -------------------------------------
                # 1. DATASET INFORMATION
                # -------------------------------------

                st.subheader("Dataset Information")

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Rows", custom_df.shape[0])
                col2.metric("Total Columns", custom_df.shape[1])
                col3.metric(
                    "Missing Values",
                    int(custom_df.isna().sum().sum())
                )

                # -------------------------------------
                # 2. PREVIEW UPLOADED DATA
                # -------------------------------------

                st.subheader("Preview of Uploaded Data")

                st.dataframe(
                    custom_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------
                # 3. SELECT MEASUREMENT COLUMN
                # -------------------------------------

                numeric_columns = custom_df.select_dtypes(
                    include=np.number
                ).columns.tolist()

                if not numeric_columns:

                    st.error(
                        "No numeric columns found. "
                        "Please upload a CSV containing numerical measurements."
                    )

                else:

                    signal_column = st.selectbox(
                        "Select the measurement column",
                        numeric_columns,
                        key="custom_signal_column"
                    )

                    signal_series = pd.to_numeric(
                        custom_df[signal_column],
                        errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan)

                    # -------------------------------------
                    # 4. SIGNAL SETTINGS
                    # -------------------------------------

                    st.subheader("Configure Signal Settings")

                    col1, col2 = st.columns(2)

                    with col1:

                        sampling_rate = st.number_input(
                            "Sampling frequency (Hz)",
                            min_value=1.0,
                            max_value=1000000.0,
                            value=1000.0,
                            step=100.0,
                            key="custom_sampling_rate"
                        )

                    with col2:

                        apply_filter = st.checkbox(
                            "Apply Butterworth low-pass filter",
                            value=True,
                            key="custom_apply_filter"
                        )

                    # -------------------------------------
                    # 5. VOLTAGE UNIT CONFIGURATION
                    # -------------------------------------

                    if analysis_type == "TENG Voltage Data":

                        voltage_unit = st.selectbox(
                            "Select voltage unit",
                            ["V", "mV"],
                            key="custom_voltage_unit"
                        )

                    else:
                        voltage_unit = "N"

                    # -------------------------------------
                    # 6. VALIDATE AND PREPARE SIGNAL
                    # -------------------------------------

                    valid_signal = signal_series.dropna()

                    if len(valid_signal) < 20:

                        st.error(
                            "The selected measurement column contains "
                            "fewer than 20 valid samples."
                        )

                    else:

                        signal = valid_signal.to_numpy(dtype=float)

                        if analysis_type == "TENG Voltage Data":

                            if voltage_unit == "mV":
                                signal = signal / 1000.0

                        filtered_signal = signal.copy()

                        # -------------------------------------
                        # 7. SIGNAL FILTERING
                        # -------------------------------------

                        if apply_filter:

                            nyquist_frequency = sampling_rate / 2.0

                            cutoff = st.number_input(
                                "Cutoff frequency (Hz)",
                                min_value=0.1,
                                max_value=float(nyquist_frequency * 0.99),
                                value=float(
                                    min(20.0, nyquist_frequency * 0.8)
                                ),
                                step=1.0,
                                key="custom_cutoff_frequency"
                            )

                            try:

                                sos = butter(
                                    4,
                                    cutoff,
                                    btype="lowpass",
                                    fs=sampling_rate,
                                    output="sos"
                                )

                                filtered_signal = sosfiltfilt(
                                    sos,
                                    signal
                                )

                                st.success(
                                    "Butterworth filtering completed successfully."
                                )

                            except ValueError as error:

                                filtered_signal = signal.copy()

                                st.warning(
                                    f"Filtering could not be completed: {error}. "
                                    "The original signal will be used."
                                )

                        # -------------------------------------
                        # 8. GENERATE TIME AXIS
                        # -------------------------------------

                        time_axis = (
                            np.arange(len(signal)) / sampling_rate
                        )


                    # -------------------------------------
                    # 4. SIGNAL VISUALISATION
                    # -------------------------------------

                    st.subheader("Signal Visualisation")

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=time_axis,
                            y=signal,
                            name="Raw Signal"
                        )
                    )

                    if apply_filter:

                        fig.add_trace(
                            go.Scatter(
                                x=time_axis,
                                y=filtered_signal,
                                name="Filtered Signal"
                            )
                        )

                    fig.update_layout(
                        title="Uploaded Experimental Signal",
                        xaxis_title="Time (seconds)",
                        yaxis_title=(
                            "Force (N)"
                            if analysis_type == "Force-Plate Gait Data"
                            else "Voltage (V)"
                        ),
                        height=500
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    # -------------------------------------
                    # 5. FORCE-PLATE ANALYSIS
                    # -------------------------------------

                    if analysis_type == "Force-Plate Gait Data":

                        st.subheader("Individual Gait Analysis")

                        threshold = st.number_input(
                            "Contact detection threshold (N)",
                            min_value=0.0,
                            value=20.0,
                            step=5.0
                        )

                        minimum_duration = st.number_input(
                            "Minimum contact duration (seconds)",
                            min_value=0.0,
                            value=0.10,
                            step=0.05
                        )

                        # Detect continuous contact intervals
                        contact_mask = filtered_signal > threshold

                        transitions = np.diff(
                            np.r_[False, contact_mask, False].astype(int)
                        )

                        starts = np.where(transitions == 1)[0]
                        ends = np.where(transitions == -1)[0]

                        gait_results = []

                        for start, end in zip(starts, ends):

                            duration = (end - start) / sampling_rate

                            if duration < minimum_duration:
                                continue

                            segment = filtered_signal[start:end]

                            if len(segment) == 0:
                                continue

                            gait_results.append({
                                "Contact Start (s)": start / sampling_rate,
                                "Contact End (s)": end / sampling_rate,
                                "Stance Duration (s)": duration,
                                "Peak Force (N)": np.max(segment),
                                "Mean Force (N)": np.mean(segment),
                                "Force Std (N)": np.std(segment),
                                "Force Impulse (N.s)": np.trapz(
                                    segment,
                                    dx=1 / sampling_rate
                                )
                            })

                        if gait_results:

                            gait_df = pd.DataFrame(gait_results)

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Detected Contacts",
                                len(gait_df)
                            )

                            col2.metric(
                                "Average Stance Duration",
                                f"{gait_df['Stance Duration (s)'].mean():.3f} s"
                            )

                            col3.metric(
                                "Average Peak Force",
                                f"{gait_df['Peak Force (N)'].mean():.2f} N"
                            )

                            st.subheader("Detected Contact Results")

                            st.dataframe(
                                gait_df.round(4),
                                use_container_width=True
                            )

                            st.download_button(
                                "Download Gait Results",
                                data=gait_df.to_csv(index=False),
                                file_name="custom_gait_results.csv",
                                mime="text/csv"
                            )

                        else:

                            st.warning(
                                "No contact intervals were detected. "
                                "Check your selected signal, threshold "
                                "and sampling frequency."
                            )

                    # -------------------------------------
                    # 6. TENG VOLTAGE ANALYSIS
                    # -------------------------------------

                    elif analysis_type == "TENG Voltage Data":

                        st.subheader("Individual TENG Signal Analysis")

                        raw_mean = np.mean(signal)
                        raw_std = np.std(signal)

                        voltage_max = np.max(filtered_signal)
                        voltage_min = np.min(filtered_signal)

                        peak_to_peak = (
                            voltage_max - voltage_min
                        )

                        rms_voltage = np.sqrt(
                            np.mean(filtered_signal ** 2)
                        )

                        col1, col2, col3, col4 = st.columns(4)

                        col1.metric(
                            "Maximum Voltage",
                            f"{voltage_max:.3f} V"
                        )

                        col2.metric(
                            "Minimum Voltage",
                            f"{voltage_min:.3f} V"
                        )

                        col3.metric(
                            "Peak-to-Peak Voltage",
                            f"{peak_to_peak:.3f} V"
                        )

                        col4.metric(
                            "RMS Voltage",
                            f"{rms_voltage:.3f} V"
                        )

                        st.write(
                            f"Raw signal mean: {raw_mean:.4f} V"
                        )

                        st.write(
                            f"Raw signal standard deviation: {raw_std:.4f} V"
                        )

                        teng_results = pd.DataFrame({
                            "Metric": [
                                "Mean Voltage",
                                "Standard Deviation",
                                "Maximum Voltage",
                                "Minimum Voltage",
                                "Peak-to-Peak Voltage",
                                "RMS Voltage"
                            ],
                            "Value (V)": [
                                np.mean(filtered_signal),
                                np.std(filtered_signal),
                                voltage_max,
                                voltage_min,
                                peak_to_peak,
                                rms_voltage
                            ]
                        })

                        st.subheader("TENG Signal Statistics")

                        st.dataframe(
                            teng_results.round(5),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download TENG Results",
                            data=teng_results.to_csv(index=False),
                            file_name="custom_teng_results.csv",
                            mime="text/csv"
                        )

        except Exception as error:

            st.error(
                f"Unable to process this CSV file: {error}"
            )

    else:

        st.info(
            "Upload a CSV file to begin your individual analysis."
        )


st.divider()

st.caption(
    "TENG-Gait Analytics | Research and Educational Prototype"
)
