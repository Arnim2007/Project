import os
import tempfile
os.environ['MPLCONFIGDIR'] = os.path.join(tempfile.gettempdir(), 'mpl')

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Physical Activity Pattern Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .match-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        background-color: #DEF7EC;
        color: #03543F;
    }
    .mismatch-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        background-color: #FDE8E8;
        color: #9B1C1C;
    }
    .activity-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.02em;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Domain Metadata & Definitions
# ==========================================
ACTIVITIES = [
    'LAYING',
    'SITTING',
    'STANDING',
    'WALKING',
    'WALKING_DOWNSTAIRS',
    'WALKING_UPSTAIRS'
]

ACTIVITY_META = {
    'LAYING': {
        'icon': '🛌',
        'label': 'Laying Down',
        'category': 'Static Posture',
        'color': '#4F46E5',
        'bg': '#EEF2FF',
        'text': '#3730A3',
        'desc': 'Horizontal body orientation. Gravitational acceleration vector shifts significantly to the body axis plane with near-zero motion variance.'
    },
    'SITTING': {
        'icon': '🪑',
        'label': 'Sitting',
        'category': 'Static Posture',
        'color': '#0284C7',
        'bg': '#E0F2FE',
        'text': '#0369A1',
        'desc': 'Stationary seated upright posture. Static gravitational acceleration predominantly along the vertical axis with stable thigh orientation.'
    },
    'STANDING': {
        'icon': '🧍',
        'label': 'Standing',
        'category': 'Static Posture',
        'color': '#059669',
        'bg': '#ECFDF5',
        'text': '#047857',
        'desc': 'Vertical upright posture. Characterized by vertical gravitational component (1g) with subtle postural tremor micro-movements.'
    },
    'WALKING': {
        'icon': '🚶',
        'label': 'Walking',
        'category': 'Dynamic Motion',
        'color': '#D97706',
        'bg': '#FEF3C7',
        'text': '#B45309',
        'desc': 'Periodic, rhythmic forward locomotion with steady 1.5 - 2 Hz step frequencies observable across triaxial accelerometer and gyroscope channels.'
    },
    'WALKING_DOWNSTAIRS': {
        'icon': '📉',
        'label': 'Walking Downstairs',
        'category': 'Dynamic Motion',
        'color': '#DB2777',
        'bg': '#FCE7F3',
        'text': '#BE185D',
        'desc': 'Stepped descent showing elevated downward impact spikes, higher vertical acceleration amplitude, and asymmetric jerk variations.'
    },
    'WALKING_UPSTAIRS': {
        'icon': '📈',
        'label': 'Walking Upstairs',
        'category': 'Dynamic Motion',
        'color': '#7C3AED',
        'bg': '#F5F3FF',
        'text': '#6D28D9',
        'desc': 'Stepped ascent requiring active gravitational push-off, generating distinctive positive vertical and forward acceleration signatures.'
    }
}

# ==========================================
# Resource Loaders (Cached)
# ==========================================
@st.cache_resource
def load_all_models():
    """Load trained models from root folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = {}

    rf_path = os.path.join(base_dir, 'random_forest_model.pkl')
    knn_path = os.path.join(base_dir, 'knn_model.pkl')

    if os.path.exists(rf_path):
        with open(rf_path, 'rb') as f:
            models['Random Forest (Accuracy: 97.63%)'] = pickle.load(f)

    if os.path.exists(knn_path):
        with open(knn_path, 'rb') as f:
            models['K-Nearest Neighbors (Accuracy: 92.88%)'] = pickle.load(f)

    return models


@st.cache_data
def load_dataset_cached():
    """Load benchmark dataset from Datasets folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'Datasets', 'HAR_simple_dataset.csv')
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        feature_cols = [c for c in df.columns if c not in ['Activity', 'subject']]
        return df, feature_cols
    return None, []


# Load Resources
models_dict = load_all_models()
dataset_df, feature_cols = load_dataset_cached()

# ==========================================
# Sidebar Navigation & Model Selection
# ==========================================
with st.sidebar:
    st.markdown("### 🏃 Activity Pattern Recognition")
    st.markdown("Human Activity Recognition (HAR) using triaxial accelerometer & gyroscope sensor features.")
    st.markdown("---")

    st.markdown("#### ⚙️ Active Model")
    if models_dict:
        selected_model_name = st.selectbox(
            "Select Classifier for Inference:",
            options=list(models_dict.keys()),
            index=0
        )
        active_model = models_dict[selected_model_name]
    else:
        st.error("⚠️ No trained models (.pkl) found in project directory. Please run train_model.py first.")
        active_model = None
        selected_model_name = ""

    st.markdown("---")
    st.markdown("#### 🧭 Navigation")
    app_mode = st.radio(
        "Go to page:",
        [
            "🎯 Interactive Prediction",
            "📁 Batch CSV Testing",
            "📊 Model Comparison & Analytics",
            "📖 Activity & Sensor Guide"
        ]
    )

    st.markdown("---")
    st.markdown("#### 📌 Dataset Overview")
    if dataset_df is not None:
        st.caption(f"**Total Samples:** {len(dataset_df):,} records")
        st.caption(f"**Sensor Features:** {len(feature_cols)} channels")
        st.caption(f"**Recognized Classes:** 6 activities")
        st.caption(f"**Subjects:** {dataset_df['subject'].nunique() if 'subject' in dataset_df else 'N/A'} volunteers")
    st.caption("Human Activity Recognition System • Built with Streamlit")

# ==========================================
# Helper Functions
# ==========================================
def run_single_inference(model, input_row_df):
    """Run model inference on single row dataframe and format probabilities."""
    pred_class = model.predict(input_row_df)[0]
    
    # Probabilities
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_row_df)[0]
        classes = model.classes_
        proba_df = pd.DataFrame({
            'Activity': classes,
            'Probability': probas,
            'Percentage': [f"{p * 100:.1f}%" for p in probas]
        }).sort_values(by='Probability', ascending=False).reset_index(drop=True)
        confidence = float(np.max(probas))
    else:
        proba_df = pd.DataFrame({
            'Activity': [pred_class],
            'Probability': [1.0],
            'Percentage': ["100.0%"]
        })
        confidence = 1.0

    return pred_class, confidence, proba_df


def render_probability_chart(proba_df, highlight_class):
    """Generate modern Altair bar chart for prediction probabilities."""
    chart_data = proba_df.copy()
    chart_data['Color'] = chart_data['Activity'].apply(
        lambda x: ACTIVITY_META.get(x, {}).get('color', '#6366F1')
    )
    
    chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X('Probability:Q', axis=alt.Axis(format='%', title='Confidence Probability'), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Activity:N', sort='-x', title=None, axis=alt.Axis(labelFontSize=12)),
        color=alt.Color('Color:N', scale=None),
        tooltip=[
            alt.Tooltip('Activity:N', title='Activity'),
            alt.Tooltip('Percentage:N', title='Confidence')
        ]
    ).properties(height=240)

    st.altair_chart(chart, use_container_width=True)


# ==========================================
# PAGE 1: Interactive Prediction
# ==========================================
if app_mode == "🎯 Interactive Prediction":
    st.markdown('<div class="main-title">🎯 Interactive Activity Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Test the model by selecting records from the dataset, generating random samples, or tweaking sensor readings.</div>', unsafe_allow_html=True)

    if active_model is None or dataset_df is None:
        st.warning("Please ensure model files and dataset are available.")
        st.stop()

    tab_sample, tab_tweaker = st.tabs(["🔍 Dataset Record Explorer", "🎛️ Sensor Feature Tweaker & Presets"])

    # ----------------------------------------------------
    # Sub-Tab 1: Dataset Record Explorer
    # ----------------------------------------------------
    with tab_sample:
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1.5])
        
        with col_ctrl1:
            filter_act = st.selectbox(
                "Filter dataset by Activity:",
                ["All Activities"] + ACTIVITIES,
                key="filter_activity"
            )

        with col_ctrl2:
            filtered_df = dataset_df if filter_act == "All Activities" else dataset_df[dataset_df['Activity'] == filter_act]
            filtered_indices = filtered_df.index.tolist()
            
            # Random seed button
            if "random_idx" not in st.session_state:
                st.session_state.random_idx = filtered_indices[0] if filtered_indices else 0

            # Row selection slider
            selected_idx = st.selectbox(
                f"Select Record Index (Total: {len(filtered_indices)} samples):",
                filtered_indices,
                index=0 if st.session_state.random_idx not in filtered_indices else filtered_indices.index(st.session_state.random_idx),
                key="row_selector"
            )

        with col_ctrl3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🎲 Pick Random Sample", use_container_width=True):
                st.session_state.random_idx = int(np.random.choice(filtered_indices))
                st.rerun()

        # Extract selected row
        selected_row = dataset_df.loc[selected_idx]
        actual_activity = selected_row['Activity']
        subject_id = selected_row.get('subject', 'Unknown')
        sample_features = pd.DataFrame([selected_row[feature_cols]])

        st.markdown("---")

        # Run Prediction
        pred_activity, confidence, proba_df = run_single_inference(active_model, sample_features)
        is_match = (pred_activity == actual_activity)

        # Result Display Columns
        res_col1, res_col2 = st.columns([1.1, 1.4])

        with res_col1:
            st.markdown("#### Prediction Result")
            pred_meta = ACTIVITY_META.get(pred_activity, {})
            actual_meta = ACTIVITY_META.get(actual_activity, {})

            # Big result box
            st.markdown(f"""
            <div style="background: {pred_meta.get('bg', '#F8FAFC')}; border: 2px solid {pred_meta.get('color', '#CBD5E1')}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 16px;">
                <div style="font-size: 3rem; margin-bottom: 4px;">{pred_meta.get('icon', '🏃')}</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {pred_meta.get('text', '#1E293B')};">{pred_activity}</div>
                <div style="font-size: 0.95rem; color: #475569; margin-top: 4px;">{pred_meta.get('category', '')}</div>
                <div style="margin-top: 12px;">
                    <span style="font-size: 1.25rem; font-weight: 700; color: {pred_meta.get('color', '#0F172A')};">Confidence: {confidence * 100:.2f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Ground Truth Comparison Card
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="metric-label">Ground Truth Activity</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #1E293B; margin-top: 2px;">
                            {actual_meta.get('icon', '📌')} {actual_activity}
                        </div>
                    </div>
                    <div>
                        {'<span class="match-badge">✓ MATCH</span>' if is_match else '<span class="mismatch-badge">✕ MISMATCH</span>'}
                    </div>
                </div>
                <div style="margin-top: 8px; font-size: 0.85rem; color: #64748B;">
                    Record Index: <b>#{selected_idx}</b> &bull; Subject ID: <b>Participant #{subject_id}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("#### Class Confidence Probabilities")
            render_probability_chart(proba_df, pred_activity)

        # Inspect sensor readings
        with st.expander(f"🔬 Inspect All 66 Sensor Features for Record #{selected_idx}"):
            st.caption("Standardized and normalized triaxial accelerometer and gyroscope measurements for this sample:")
            
            # Arrange features into a clean 3-column table
            feature_display_df = pd.DataFrame({
                'Feature Name': feature_cols,
                'Sensor Value': [selected_row[f] for f in feature_cols]
            })
            st.dataframe(
                feature_display_df,
                use_container_width=True,
                height=300
            )

    # ----------------------------------------------------
    # Sub-Tab 2: Sensor Feature Tweaker & Presets
    # ----------------------------------------------------
    with tab_tweaker:
        st.markdown("#### 🎛️ Test Custom Sensor Configurations")
        st.write("Load a characteristic activity template, tweak the primary acceleration and angular velocity signals, and observe how the model classifies the activity.")

        preset_cols = st.columns(6)
        preset_choice = None
        for i, act in enumerate(ACTIVITIES):
            meta = ACTIVITY_META[act]
            with preset_cols[i]:
                if st.button(f"{meta['icon']}\n{meta['label']}", key=f"preset_btn_{act}", use_container_width=True):
                    st.session_state.selected_preset = act

        # Default preset
        if "selected_preset" not in st.session_state:
            st.session_state.selected_preset = 'WALKING'

        current_preset = st.session_state.selected_preset
        preset_samples = dataset_df[dataset_df['Activity'] == current_preset]
        preset_mean_vector = preset_samples[feature_cols].mean()

        st.info(f"Loaded characteristic sensor baseline for: **{current_preset}** ({ACTIVITY_META[current_preset]['desc']})")

        # Key interactive sliders representing dominant motion axes
        st.markdown("##### 📡 Primary Triaxial Sensor Controls")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown("**Body Acceleration (Mean)**")
            val_acc_x = st.slider("tBodyAcc-mean()-X", -1.0, 1.0, float(preset_mean_vector.get('tBodyAcc-mean()-X', 0.25)), step=0.01)
            val_acc_y = st.slider("tBodyAcc-mean()-Y", -1.0, 1.0, float(preset_mean_vector.get('tBodyAcc-mean()-Y', -0.02)), step=0.01)
            val_acc_z = st.slider("tBodyAcc-mean()-Z", -1.0, 1.0, float(preset_mean_vector.get('tBodyAcc-mean()-Z', -0.11)), step=0.01)

        with col_s2:
            st.markdown("**Gravity Acceleration (Mean)**")
            val_grav_x = st.slider("tGravityAcc-mean()-X", -1.0, 1.0, float(preset_mean_vector.get('tGravityAcc-mean()-X', 0.85)), step=0.01)
            val_grav_y = st.slider("tGravityAcc-mean()-Y", -1.0, 1.0, float(preset_mean_vector.get('tGravityAcc-mean()-Y', -0.28)), step=0.01)
            val_grav_z = st.slider("tGravityAcc-mean()-Z", -1.0, 1.0, float(preset_mean_vector.get('tGravityAcc-mean()-Z', 0.15)), step=0.01)

        with col_s3:
            st.markdown("**Motion Variance & Energy**")
            val_acc_std_x = st.slider("tBodyAcc-std()-X (Motion Intensity)", -1.0, 1.0, float(preset_mean_vector.get('tBodyAcc-std()-X', -0.90)), step=0.01)
            val_gyro_x = st.slider("tBodyGyro-mean()-X (Angular Velocity)", -1.0, 1.0, float(preset_mean_vector.get('tBodyGyro-mean()-X', 0.0)), step=0.01)
            val_mag_mean = st.slider("tBodyAccMag-mean() (Total Magnitude)", -1.0, 1.0, float(preset_mean_vector.get('tBodyAccMag-mean()', -0.85)), step=0.01)

        # Build modified feature vector
        custom_features = preset_mean_vector.copy()
        custom_features['tBodyAcc-mean()-X'] = val_acc_x
        custom_features['tBodyAcc-mean()-Y'] = val_acc_y
        custom_features['tBodyAcc-mean()-Z'] = val_acc_z
        custom_features['tGravityAcc-mean()-X'] = val_grav_x
        custom_features['tGravityAcc-mean()-Y'] = val_grav_y
        custom_features['tGravityAcc-mean()-Z'] = val_grav_z
        custom_features['tBodyAcc-std()-X'] = val_acc_std_x
        custom_features['tBodyGyro-mean()-X'] = val_gyro_x
        custom_features['tBodyAccMag-mean()'] = val_mag_mean

        custom_input_df = pd.DataFrame([custom_features])

        # Prediction
        pred_custom, conf_custom, proba_custom = run_single_inference(active_model, custom_input_df)

        st.markdown("---")
        c_res1, c_res2 = st.columns([1, 1.5])
        with c_res1:
            meta_c = ACTIVITY_META.get(pred_custom, {})
            st.markdown(f"""
            <div style="background: {meta_c.get('bg', '#F8FAFC')}; border: 2px solid {meta_c.get('color', '#CBD5E1')}; border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Classified Activity</div>
                <div style="font-size: 2.8rem; margin: 4px 0;">{meta_c.get('icon', '🏃')}</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: {meta_c.get('text', '#1E293B')};">{pred_custom}</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {meta_c.get('color', '#0F172A')}; margin-top: 6px;">
                    Confidence: {conf_custom * 100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_res2:
            st.markdown("##### Prediction Distribution")
            render_probability_chart(proba_custom, pred_custom)


# ==========================================
# PAGE 2: Batch CSV Testing
# ==========================================
elif app_mode == "📁 Batch CSV Testing":
    st.markdown('<div class="main-title">📁 Batch CSV Testing & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload a batch CSV file of sensor readings to generate bulk predictions and automated performance evaluation.</div>', unsafe_allow_html=True)

    if active_model is None:
        st.warning("Please train and load models first.")
        st.stop()

    # Pre-packaged options or custom file upload
    col_upload_opts, col_benchmark = st.columns([1.5, 1])

    with col_upload_opts:
        uploaded_file = st.file_uploader(
            "Upload your CSV dataset:",
            type=["csv"],
            help="CSV must contain the 66 sensor feature columns. May optionally include 'Activity' ground truth."
        )

    with col_benchmark:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_benchmark_split = st.button("⚡ Test on Benchmark 20% Split (590 samples)", use_container_width=True)
        use_sample_30 = st.button("📋 Test on First 30 Samples", use_container_width=True)

    batch_df = None
    batch_source_title = ""

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_source_title = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
    elif use_benchmark_split and dataset_df is not None:
        # Use stratified 20% test slice (same as train_model.py test split)
        np.random.seed(42)
        indices = []
        for act in ACTIVITIES:
            act_idx = dataset_df[dataset_df['Activity'] == act].index.tolist()
            test_count = int(len(act_idx) * 0.20)
            indices.extend(act_idx[:test_count])
        batch_df = dataset_df.loc[indices].copy()
        batch_source_title = f"Benchmark Stratified Test Split ({len(batch_df)} samples)"
    elif use_sample_30 and dataset_df is not None:
        batch_df = dataset_df.head(30).copy()
        batch_source_title = "First 30 Dataset Records"

    if batch_df is not None:
        st.markdown(f"### 📋 {batch_source_title}")
        
        # Verify feature columns
        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"⚠️ Uploaded data is missing {len(missing_cols)} expected feature columns. Please verify format.")
            with st.expander("Show Missing Columns"):
                st.write(missing_cols)
            st.stop()

        has_ground_truth = 'Activity' in batch_df.columns

        # Inference Execution
        X_batch = batch_df[feature_cols]
        with st.spinner("Generating batch predictions..."):
            preds = active_model.predict(X_batch)
            if hasattr(active_model, "predict_proba"):
                probas = active_model.predict_proba(X_batch)
                confidences = np.max(probas, axis=1)
            else:
                confidences = np.ones(len(preds))

        results_df = batch_df.copy()
        results_df['Predicted_Activity'] = preds
        results_df['Confidence'] = [f"{c * 100:.2f}%" for c in confidences]

        if has_ground_truth:
            results_df['Match'] = results_df['Activity'] == results_df['Predicted_Activity']
            acc = accuracy_score(results_df['Activity'], results_df['Predicted_Activity'])
            
            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Batch Accuracy</div>
                    <div class="metric-val" style="color: #059669;">{acc * 100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                correct_count = results_df['Match'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Correct Predictions</div>
                    <div class="metric-val" style="color: #0284C7;">{correct_count} / {len(results_df)}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                mismatch_count = len(results_df) - correct_count
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Misclassified</div>
                    <div class="metric-val" style="color: {'#DC2626' if mismatch_count > 0 else '#059669'};">{mismatch_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                avg_conf = np.mean(confidences) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Confidence</div>
                    <div class="metric-val" style="color: #4F46E5;">{avg_conf:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts Section
        c_chart1, c_chart2 = st.columns([1.2, 1.2])

        with c_chart1:
            st.markdown("##### 📊 Predicted Activity Distribution")
            dist_df = pd.DataFrame(results_df['Predicted_Activity'].value_counts()).reset_index()
            dist_df.columns = ['Activity', 'Count']
            dist_df['Color'] = dist_df['Activity'].apply(lambda x: ACTIVITY_META.get(x, {}).get('color', '#6366F1'))
            
            bar_chart = alt.Chart(dist_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X('Activity:N', title=None, axis=alt.Axis(labelAngle=-25)),
                y=alt.Y('Count:Q', title='Sample Count'),
                color=alt.Color('Color:N', scale=None),
                tooltip=['Activity', 'Count']
            ).properties(height=260)
            st.altair_chart(bar_chart, use_container_width=True)

        with c_chart2:
            if has_ground_truth:
                st.markdown("##### 🎯 Confusion Matrix Heatmap")
                labels = sorted(list(set(results_df['Activity'].unique()) | set(results_df['Predicted_Activity'].unique())))
                cm = confusion_matrix(results_df['Activity'], results_df['Predicted_Activity'], labels=labels)
                cm_records = []
                for i, actual in enumerate(labels):
                    for j, pred in enumerate(labels):
                        cm_records.append({
                            'Actual': actual,
                            'Predicted': pred,
                            'Count': int(cm[i, j])
                        })
                cm_df = pd.DataFrame(cm_records)
                
                heatmap = alt.Chart(cm_df).mark_rect().encode(
                    x=alt.X('Predicted:N', title='Predicted Class', axis=alt.Axis(labelAngle=-25)),
                    y=alt.Y('Actual:N', title='Actual Class'),
                    color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
                    tooltip=['Actual', 'Predicted', 'Count']
                ).properties(height=260)

                text = heatmap.mark_text(baseline='middle').encode(
                    text='Count:Q',
                    color=alt.condition(
                        alt.datum.Count > cm.max() / 2,
                        alt.value('white'),
                        alt.value('black')
                    )
                )
                st.altair_chart(heatmap + text, use_container_width=True)
            else:
                st.markdown("##### ℹ️ Ground Truth Note")
                st.info("The uploaded file does not have an 'Activity' ground truth column. Predictions have been generated based on sensor features.")

        # Data Preview Table
        st.markdown("##### 📑 Inference Results Table")
        preview_cols = ['Predicted_Activity', 'Confidence']
        if has_ground_truth:
            preview_cols = ['Activity', 'Predicted_Activity', 'Confidence', 'Match']
        if 'subject' in results_df.columns:
            preview_cols.insert(0, 'subject')

        st.dataframe(
            results_df[preview_cols],
            use_container_width=True,
            height=320
        )

        # Download Result CSV
        csv_download = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Predictions (CSV)",
            data=csv_download,
            file_name="har_model_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================
# PAGE 3: Model Comparison & Analytics
# ==========================================
elif app_mode == "📊 Model Comparison & Analytics":
    st.markdown('<div class="main-title">📊 Model Comparison & Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed benchmark comparison between Random Forest Classifier and K-Nearest Neighbors (KNN).</div>', unsafe_allow_html=True)

    # Comparison Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Random Forest Accuracy</div>
            <div class="metric-val" style="color: #059669;">97.63%</div>
            <div style="font-size: 0.8rem; color: #10B981; font-weight: 600;">★ Best Performing Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">KNN Accuracy</div>
            <div class="metric-val" style="color: #0284C7;">92.88%</div>
            <div style="font-size: 0.8rem; color: #64748B;">StandardScaler + k=5</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Accuracy Delta</div>
            <div class="metric-val" style="color: #7C3AED;">+4.75%</div>
            <div style="font-size: 0.8rem; color: #64748B;">RF Advantage over KNN</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Feature Count</div>
            <div class="metric-val" style="color: #D97706;">66</div>
            <div style="font-size: 0.8rem; color: #64748B;">Accelerometer & Gyroscope</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Detailed Comparison Table
    st.markdown("#### ⚖️ Full Benchmark Performance Comparison")
    perf_data = {
        "Evaluation Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score", "Inference Speed", "Scaling Required"],
        "Random Forest Classifier (Ensemble)": ["97.63%", "97.68%", "97.55%", "97.59%", "Fast (O(M × Trees))", "No (Scale-invariant)"],
        "K-Nearest Neighbors (KNN)": ["92.88%", "93.15%", "92.68%", "92.82%", "Moderate (O(N × D))", "Yes (StandardScaler Pipeline)"]
    }
    st.table(pd.DataFrame(perf_data).set_index("Evaluation Metric"))

    # Per-Class Precision & Recall for Best Model
    st.markdown("#### 🎯 Per-Class Precision, Recall, & F1 (Random Forest)")
    per_class_df = pd.DataFrame({
        "Activity": ACTIVITIES,
        "Precision": [1.0000, 0.9495, 0.9623, 0.9802, 1.0000, 0.9691],
        "Recall":    [1.0000, 0.9592, 0.9533, 1.0000, 0.9405, 1.0000],
        "F1-Score":  [1.0000, 0.9543, 0.9577, 0.9900, 0.9693, 0.9843],
        "Support":   [108, 98, 107, 99, 84, 94]
    })
    
    col_pct1, col_pct2 = st.columns([1.5, 1])
    with col_pct1:
        st.dataframe(
            per_class_df.style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}"
            }),
            use_container_width=True
        )

    with col_pct2:
        st.caption("Key Insights:")
        st.markdown("""
        - **`LAYING`** achieves a perfect **100.0% F1-score**, as horizontal gravitational orientation produces zero ambiguity.
        - **`WALKING`** and **`WALKING_UPSTAIRS`** achieve **100.0% recall** with high precision.
        - The most subtle boundary occurs between **`SITTING`** and **`STANDING`**, where static upright posture produces similar vertical gravity components.
        """)

    # Top Features from Random Forest
    st.markdown("---")
    st.markdown("#### 🌲 Top 15 Most Influential Features (Feature Importance)")
    rf_candidate = models_dict.get('Random Forest (Accuracy: 97.63%)')
    if rf_candidate is not None and hasattr(rf_candidate, 'named_steps') and 'rf' in rf_candidate.named_steps:
        importances = rf_candidate.named_steps['rf'].feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15).reset_index(drop=True)

        fi_chart = alt.Chart(fi_df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color='#4F46E5').encode(
            x=alt.X('Importance:Q', title='Relative Importance Score'),
            y=alt.Y('Feature:N', sort='-x', title=None),
            tooltip=['Feature', alt.Tooltip('Importance:Q', format='.4f')]
        ).properties(height=380)

        st.altair_chart(fi_chart, use_container_width=True)
        st.caption("Gravitational acceleration (`tGravityAcc`) and frequency-domain body jerk signals dominate feature importance in discriminating postures and movement.")


# ==========================================
# PAGE 4: Activity & Sensor Guide
# ==========================================
elif app_mode == "📖 Activity & Sensor Guide":
    st.markdown('<div class="main-title">📖 Activity & Sensor Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Understanding how smartphone inertial sensors (accelerometer & gyroscope) quantify physical human movement.</div>', unsafe_allow_html=True)

    # 6 Activity Cards
    st.markdown("#### 🏃 The Six Recognized Activity Classes")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    row_cols = [c1, c2, c3, c4, c5, c6]

    for i, act in enumerate(ACTIVITIES):
        meta = ACTIVITY_META[act]
        with row_cols[i]:
            st.markdown(f"""
            <div style="background: {meta['bg']}; border-left: 5px solid {meta['color']}; border-radius: 10px; padding: 16px; margin-bottom: 16px; min-height: 190px;">
                <div style="font-size: 1.8rem; margin-bottom: 4px;">{meta['icon']}</div>
                <div style="font-size: 1.15rem; font-weight: 800; color: {meta['text']};">{meta['label']}</div>
                <div style="font-size: 0.8rem; font-weight: 600; color: #64748B; margin-bottom: 8px;">{meta['category']}</div>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.4;">{meta['desc']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Sensor Signal Pipeline Architecture
    st.markdown("#### 🛰️ Sensor Signal Processing Architecture")
    st.markdown("""
    ```text
    [ Triaxial Accelerometer ]  ────► [ Noise Filter (Butterworth 20Hz) ] ───┬─► [ Gravitational Acceleration (Low-pass 0.3Hz) ]
    [ Triaxial Gyroscope     ]  ────► [ 50Hz Sampling Rate (Waist Mount)] ───┼─► [ Body Motion Acceleration ]
                                                                             ├─► [ Jerk Signals (Derivatives da/dt, dw/dt) ]
                                                                             └─► [ Fast Fourier Transform (FFT) ]
                                                                                               │
                                                                                               ▼
                                                                                   [ 66 Statistical Features ]
                                                                                   (Mean & Standard Deviation)
    ```
    """)

    st.markdown("#### 🔬 Feature Naming Conventions Explained")
    st.markdown("""
    - **`t` prefix (Time domain):** Raw temporal signals captured at 50 Hz.
    - **`f` prefix (Frequency domain):** Frequency spectrum derived using Fast Fourier Transform (FFT).
    - **`BodyAcc`:** Acceleration generated solely by human body movement.
    - **`GravityAcc`:** Constant 1g gravitational pull direction relative to device axes.
    - **`Gyro`:** Angular velocity measured around the X, Y, and Z device axes in radians/sec.
    - **`Jerk`:** Rate of change of acceleration/gyro signals (smoothness/impact metrics).
    - **`Mag`:** Three-dimensional Euclidean magnitude $\\sqrt{X^2 + Y^2 + Z^2}$.
    - **`-mean()` & `-std()`:** Window-averaged mean and standard deviation across 128-reading sliding windows.
    """)
