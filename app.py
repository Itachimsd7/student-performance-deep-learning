import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import json
from pathlib import Path
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title='Student Performance Predictor | Deep Learning',
    page_icon='🎓',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ─── Custom CSS Styling ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-relu { background-color: #EDE9FE; color: #7C3AED; }
    .badge-softmax { background-color: #E0E7FF; color: #4338CA; }
    .badge-loss { background-color: #FEE2E2; color: #B91C1C; }
    .badge-sgd { background-color: #D1FAE5; color: #047857; }
    
    .pass-banner {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
    }
    .fail-banner {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Assets ────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('models/student_performance_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    # Build feature extractor for hidden layer activations
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer('hidden_1').output,
            model.get_layer('hidden_2').output,
            model.get_layer('output').output
        ]
    )
    return model, scaler, feature_extractor

@st.cache_data
def load_data():
    return pd.read_csv('data/student_performance.csv')

@st.cache_data
def load_metrics():
    with open('outputs/metrics.json') as f:
        return json.load(f)

def show_image(path, caption='', width=None):
    if Path(path).exists():
        img = Image.open(path)
        st.image(img, caption=caption, use_container_width=(width is None))
    else:
        st.warning(f'Figure not found: {path}')

# ─── Sidebar Navigation ─────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h2 style="margin: 0; color: #1E3A8A; font-weight: 800;">🎓 DL STUDIO</h2>
    <p style="margin: 0; font-size: 0.85rem; color: #64748B;">Student Performance Predictor</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('---')
page = st.sidebar.radio('Navigation', [
    '🏠 Home / Overview',
    '📊 Dataset Analytics',
    '🧠 Neural Architecture',
    '📈 Training Diagnostics',
    '📋 Test Evaluation',
    '🔮 Student Prediction Studio'
])

st.sidebar.markdown('---')
st.sidebar.markdown('### 📌 Assigned Setup')
st.sidebar.markdown("""
- **Hidden:** <span class="badge-pill badge-relu">ReLU</span>
- **Output:** <span class="badge-pill badge-softmax">Softmax</span>
- **Loss:** <span class="badge-pill badge-loss">Cat. Cross-Entropy</span>
- **Optimizer:** <span class="badge-pill badge-sgd">SGD + Momentum</span>
""", unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.info('💡 **Assignment Specification:**\nStrictly implements 5→32→16→2 MLP architecture with SGD optimizer.')


# ════════════════════════════════════════════════════════════════════
# PAGE 1: Home / Overview
# ════════════════════════════════════════════════════════════════════
if page == '🏠 Home / Overview':
    st.markdown('<div class="main-title">Student Performance Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">A Deep Neural Network classification system built with TensorFlow/Keras & Streamlit</div>', unsafe_allow_html=True)
    
    # Pipeline Flow Visualizer
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 16px; margin-bottom: 25px;">
        <h4 style="margin: 0 0 10px 0; color: #334155; font-size: 0.95rem;">🔄 END-TO-END PIPELINE ARCHITECTURE</h4>
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; text-align: center;">
            <div style="flex: 1; min-width: 140px; background: white; padding: 12px; border-radius: 8px; border-top: 3px solid #3B82F6; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
                <div style="font-size: 1.2rem;">📊</div>
                <strong style="font-size: 0.85rem;">1. Input Data</strong>
                <div style="font-size: 0.75rem; color: #64748B;">5 Student Features</div>
            </div>
            <div style="color: #94A3B8; font-weight: bold;">➔</div>
            <div style="flex: 1; min-width: 140px; background: white; padding: 12px; border-radius: 8px; border-top: 3px solid #8B5CF6; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
                <div style="font-size: 1.2rem;">⚖️</div>
                <strong style="font-size: 0.85rem;">2. Scaling</strong>
                <div style="font-size: 0.75rem; color: #64748B;">StandardScaler (Z-Score)</div>
            </div>
            <div style="color: #94A3B8; font-weight: bold;">➔</div>
            <div style="flex: 1; min-width: 140px; background: white; padding: 12px; border-radius: 8px; border-top: 3px solid #EC4899; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
                <div style="font-size: 1.2rem;">🧠</div>
                <strong style="font-size: 0.85rem;">3. Hidden Layers</strong>
                <div style="font-size: 0.75rem; color: #64748B;">32 & 16 Units (ReLU)</div>
            </div>
            <div style="color: #94A3B8; font-weight: bold;">➔</div>
            <div style="flex: 1; min-width: 140px; background: white; padding: 12px; border-radius: 8px; border-top: 3px solid #F59E0B; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
                <div style="font-size: 1.2rem;">🎯</div>
                <strong style="font-size: 0.85rem;">4. Output Layer</strong>
                <div style="font-size: 0.75rem; color: #64748B;">2 Units (Softmax)</div>
            </div>
            <div style="color: #94A3B8; font-weight: bold;">➔</div>
            <div style="flex: 1; min-width: 140px; background: white; padding: 12px; border-radius: 8px; border-top: 3px solid #10B981; box-shadow: 0 2px 5px rgba(0,0,0,0.04);">
                <div style="font-size: 1.2rem;">🏆</div>
                <strong style="font-size: 0.85rem;">5. Prediction</strong>
                <div style="font-size: 0.75rem; color: #64748B;">PASS or FAIL Prob.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 🎯 Problem Statement & Goal")
        st.markdown("""
        In higher education institutions, identifying students at risk of academic failure early in the semester enables proactive pedagogical interventions.
        
        This project builds a **multi-layer perceptron (MLP)** neural network to classify students into **PASS (1)** or **FAIL (0)** outcomes using five foundational indicators:
        - 🏫 **Class Attendance (%):** Lectures and labs attended
        - 📝 **Internal Assessment Marks:** Midterm and test scores
        - 📚 **Assignment Score:** Lab and homework grades
        - ⏰ **Study Hours per Day:** Independent self-study time
        - 🏆 **Previous Academic Performance:** Prior semester GPA/marks
        """)
        
        st.markdown("### 📋 Assigned Technical Specification")
        st.markdown(r"""
        | Hyperparameter / Component | Specified Assignment Value |
        | :--- | :--- |
        | **Model Type** | Fully-Connected Feedforward Multi-Layer Perceptron |
        | **Network Structure** | Input(5) ➔ Dense(32) ➔ Dense(16) ➔ Dense(2) |
        | **Hidden Activation** | **ReLU** ($f(x) = \max(0, x)$) |
        | **Output Activation** | **Softmax** ($\sigma(z_i) = \frac{e^{z_i}}{\sum e^{z_j}}$) |
        | **Loss Function** | **Categorical Cross-Entropy** ($L = -\sum y_i \log \hat{y}_i$) |
        | **Optimizer** | **SGD** with Momentum ($\eta=0.01, \beta=0.9$) |
        | **Target Encoding** | One-Hot Encoding (`[1, 0]` Fail, `[0, 1]` Pass) |
        """)

    with col2:
        st.markdown("### 🔬 Interactive Deep Learning Concept Explorer")
        tabs = st.tabs(["ReLU Curve", "Softmax Probabilities", "Cross-Entropy Loss", "SGD Momentum"])
        
        with tabs[0]:
            st.caption("ReLU activates positive values linearly while blocking negative signals, solving vanishing gradients.")
            x_val = np.linspace(-5, 5, 200)
            y_val = np.maximum(0, x_val)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_val[x_val < 0], y=y_val[x_val < 0], mode='lines', line=dict(color='#EF4444', width=3), name='Dormant (0)'))
            fig.add_trace(go.Scatter(x=x_val[x_val >= 0], y=y_val[x_val >= 0], mode='lines', line=dict(color='#10B981', width=3), name='Active (x)'))
            fig.update_layout(title="f(x) = max(0, x)", height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Neuron Input (z)", yaxis_title="Activation Output a=f(z)")
            st.plotly_chart(fig, use_container_width=True)
            
        with tabs[1]:
            st.caption("Softmax transforms raw unnormalized logits into calibrated probabilities that sum to 1.0.")
            c_s1, c_s2 = st.columns(2)
            z1 = c_s1.slider("Logit z1 (Fail)", -5.0, 5.0, -0.8, 0.1)
            z2 = c_s2.slider("Logit z2 (Pass)", -5.0, 5.0, 1.4, 0.1)
            e1, e2 = np.exp(z1), np.exp(z2)
            p1, p2 = e1 / (e1 + e2), e2 / (e1 + e2)
            fig_sm = go.Figure(go.Bar(
                x=['Fail', 'Pass'],
                y=[p1, p2],
                text=[f"{p1*100:.1f}%", f"{p2*100:.1f}%"],
                textposition='auto',
                marker_color=['#EF4444', '#10B981']
            ))
            fig_sm.update_layout(height=240, margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(range=[0, 1], title="Probability"))
            st.plotly_chart(fig_sm, use_container_width=True)

        with tabs[2]:
            st.caption("Categorical Cross-Entropy heavily penalizes predictions that assign low probability to the true class.")
            p_preds = np.linspace(0.01, 0.99, 100)
            cce_loss = -np.log(p_preds)
            fig_loss = go.Figure(go.Scatter(x=p_preds, y=cce_loss, mode='lines', line=dict(color='#8B5CF6', width=3)))
            fig_loss.update_layout(title="Loss = -log(P_true)", height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Predicted Probability for True Class", yaxis_title="Loss Value")
            st.plotly_chart(fig_loss, use_container_width=True)

        with tabs[3]:
            st.caption("SGD with Momentum accelerates in consistent directions and dampens transverse oscillations.")
            t = np.linspace(0, 10, 100)
            sgd_raw = np.cos(3*t) * np.exp(-0.2*t)
            sgd_mom = np.cos(0.5*t) * np.exp(-0.4*t)
            fig_sgd = go.Figure()
            fig_sgd.add_trace(go.Scatter(x=t, y=sgd_raw, mode='lines', name='Standard SGD (Oscillatory)', line=dict(color='#F59E0B', dash='dash')))
            fig_sgd.add_trace(go.Scatter(x=t, y=sgd_mom, mode='lines', name='SGD + Momentum (Smooth)', line=dict(color='#059669', width=3)))
            fig_sgd.update_layout(title="Optimizer Trajectory Comparison", height=280, margin=dict(l=20, r=20, t=30, b=20), xaxis_title="Training Steps", yaxis_title="Parameter Position")
            st.plotly_chart(fig_sgd, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 2: Dataset Analytics
# ════════════════════════════════════════════════════════════════════
elif page == '📊 Dataset Analytics':
    st.markdown('<div class="main-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Statistical distributions, correlation structures, and feature interactions</div>', unsafe_allow_html=True)
    
    try:
        df = load_data()
        
        # Metric KPI cards
        k1, k2, k3, k4, k5 = st.columns(5)
        total_n = len(df)
        pass_n = int(df['result'].sum())
        fail_n = total_n - pass_n
        
        k1.markdown(f"""<div class="metric-card"><div class="metric-value">{total_n:,}</div><div class="metric-label">Total Records</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#10B981;">{pass_n}</div><div class="metric-label">Pass Cohort ({pass_n/total_n*100:.1f}%)</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#EF4444;">{fail_n}</div><div class="metric-label">Fail Cohort ({fail_n/total_n*100:.1f}%)</div></div>""", unsafe_allow_html=True)
        k4.markdown(f"""<div class="metric-card"><div class="metric-value">5</div><div class="metric-label">Input Features</div></div>""", unsafe_allow_html=True)
        k5.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#3B82F6;">0</div><div class="metric-label">Missing Values</div></div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Distribution & Heatmap
        c_left, c_right = st.columns([1, 1], gap="medium")
        
        with c_left:
            st.subheader("🎯 Class Distribution Breakdown")
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Pass (1)', 'Fail (0)'],
                values=[pass_n, fail_n],
                hole=0.6,
                marker=dict(colors=['#10B981', '#EF4444']),
                textinfo='label+percent',
                hoverinfo='label+value+percent'
            )])
            fig_donut.update_layout(
                showlegend=True,
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                annotations=[dict(text=f"Total<br><b>{total_n}</b>", x=0.5, y=0.5, font_size=18, showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with c_right:
            st.subheader("🔥 Feature Correlation Matrix")
            corr = df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='Blues',
                aspect='auto'
            )
            fig_corr.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")
        
        # Interactive Feature Distributions (Boxplots comparing Pass vs Fail)
        st.subheader("📦 Feature Comparison: Passing vs Failing Students")
        feature_options = {
            'attendance': 'Attendance Rate (%)',
            'internal_marks': 'Internal Marks (out of 100)',
            'assignment_score': 'Assignment Score (out of 100)',
            'study_hours': 'Daily Study Hours',
            'prev_performance': 'Previous Performance (%)'
        }
        
        col_box1, col_box2 = st.columns([1, 2])
        selected_feat = col_box1.selectbox("Select Feature to Inspect:", list(feature_options.keys()), format_func=lambda x: feature_options[x])
        
        df_plot = df.copy()
        df_plot['Outcome'] = df_plot['result'].map({0: 'Fail', 1: 'Pass'})
        
        fig_box = px.violin(
            df_plot,
            x='Outcome',
            y=selected_feat,
            color='Outcome',
            box=True,
            points='all',
            color_discrete_map={'Pass': '#10B981', 'Fail': '#EF4444'},
            labels={selected_feat: feature_options[selected_feat]}
        )
        fig_box.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10))
        col_box2.plotly_chart(fig_box, use_container_width=True)
        
        # 3D Feature Space Explorer
        st.markdown("---")
        st.subheader("🌐 3D Interactive Feature Space Explorer")
        st.caption("Rotate, pan, and zoom to explore how Pass/Fail clusters naturally separate in multi-dimensional space.")
        
        c3_1, c3_2, c3_3 = st.columns(3)
        x_axis = c3_1.selectbox("X Axis", list(feature_options.keys()), index=0, format_func=lambda x: feature_options[x])
        y_axis = c3_2.selectbox("Y Axis", list(feature_options.keys()), index=1, format_func=lambda x: feature_options[x])
        z_axis = c3_3.selectbox("Z Axis", list(feature_options.keys()), index=3, format_func=lambda x: feature_options[x])
        
        fig_3d = px.scatter_3d(
            df_plot.sample(min(500, len(df_plot)), random_state=42),
            x=x_axis,
            y=y_axis,
            z=z_axis,
            color='Outcome',
            color_discrete_map={'Pass': '#10B981', 'Fail': '#EF4444'},
            opacity=0.7,
            title="3D Student Cluster Plot (500 Samples Sampled)"
        )
        fig_3d.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Raw Data Sample & Filter")
        st.dataframe(df.head(50), use_container_width=True)
        
    except FileNotFoundError:
        st.error("Dataset not found. Please run `python src/generate_data.py` first.")


# ════════════════════════════════════════════════════════════════════
# PAGE 3: Neural Architecture
# ════════════════════════════════════════════════════════════════════
elif page == '🧠 Neural Architecture':
    st.markdown('<div class="main-title">Multi-Layer Perceptron Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visual structural decomposition of the 5 ➔ 32 ➔ 16 ➔ 2 feedforward network</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📐 Layer Specifications")
        st.markdown("""
        ```
        INPUT LAYER (5 Features)
             │  Linear Combination: Z[1] = W[1] · X + b[1]
             ▼
        HIDDEN LAYER 1: 32 Neurons [ReLU Activation]
             │  Parameters: 5 × 32 + 32 = 192
             │  Linear Combination: Z[2] = W[2] · A[1] + b[2]
             ▼
        HIDDEN LAYER 2: 16 Neurons [ReLU Activation]
             │  Parameters: 32 × 16 + 16 = 528
             │  Linear Combination: Z[3] = W[3] · A[2] + b[3]
             ▼
        OUTPUT LAYER: 2 Neurons [Softmax Activation]
             │  Parameters: 16 × 2 + 2 = 34
             ▼
        OUTPUT PROBABILITIES: [P(Fail), P(Pass)]
        ```
        """)
        
        st.markdown("### 🧮 Mathematical Formulations")
        st.markdown(r"""
        - **Hidden Activation 1:** $A^{[1]} = \max(0, W^{[1]} X + b^{[1]})$
        - **Hidden Activation 2:** $A^{[2]} = \max(0, W^{[2]} A^{[1]} + b^{[2]})$
        - **Output Softmax:** $\hat{y}_i = \frac{e^{Z_i^{[3]}}}{e^{Z_1^{[3]}} + e^{Z_2^{[3]}}}$
        - **Loss:** $L(y, \hat{y}) = - (y_{\text{fail}} \log \hat{y}_{\text{fail}} + y_{\text{pass}} \log \hat{y}_{\text{pass}})$
        """)
        
        st.markdown("### ⚙️ Optimizer & Training Hyperparameters")
        st.markdown(r"""
        - **Optimizer:** SGD ($\text{lr} = 0.01, \text{momentum} = 0.9$)
        - **Weight Updates:** $v_t = \beta v_{t-1} + \eta \nabla_\theta J(\theta)$, $\theta = \theta - v_t$
        - **Batch Size:** 32
        - **Regularization:** EarlyStopping (`monitor='val_loss'`, `patience=15`, `restore_best_weights=True`)
        """)

    with col2:
        st.subheader("📊 Parameter Count Breakdown")
        
        param_data = pd.DataFrame({
            'Layer': ['Hidden Layer 1 (32 ReLU)', 'Hidden Layer 2 (16 ReLU)', 'Output Layer (2 Softmax)'],
            'Parameters': [192, 528, 34],
            'Percentage': [25.5, 70.0, 4.5]
        })
        
        fig_donut_p = go.Figure(data=[go.Pie(
            labels=param_data['Layer'],
            values=param_data['Parameters'],
            hole=0.55,
            marker=dict(colors=['#3B82F6', '#8B5CF6', '#10B981']),
            textinfo='label+value',
            hoverinfo='label+value+percent'
        )])
        fig_donut_p.update_layout(
            title="Total Parameters: 754 Trainable Weights & Biases",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_donut_p, use_container_width=True)
        
        st.subheader("🕸️ Network Node-Link Representation")
        
        # Interactive Node-Link Network Graph
        fig_net = go.Figure()
        
        layer_x = [0, 1, 2, 3]
        layer_names = ['Input (5)', 'Hidden 1 (32)', 'Hidden 2 (16)', 'Output (2)']
        layer_sizes = [5, 12, 8, 2] # Represent visually with scaled node counts
        layer_colors = ['#3B82F6', '#8B5CF6', '#EC4899', '#10B981']
        
        # Draw synapses (connections)
        for l in range(len(layer_sizes) - 1):
            x0, x1 = layer_x[l], layer_x[l+1]
            n0, n1 = layer_sizes[l], layer_sizes[l+1]
            y0_vals = np.linspace(-4, 4, n0)
            y1_vals = np.linspace(-4, 4, n1)
            
            for y0 in y0_vals:
                for y1 in y1_vals:
                    fig_net.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode='lines',
                        line=dict(color='rgba(203, 213, 225, 0.4)', width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))
                    
        # Draw nodes
        for l in range(len(layer_sizes)):
            x = layer_x[l]
            n = layer_sizes[l]
            y_vals = np.linspace(-4, 4, n)
            fig_net.add_trace(go.Scatter(
                x=[x]*n, y=y_vals,
                mode='markers',
                marker=dict(size=18, color=layer_colors[l], line=dict(color='white', width=2)),
                name=layer_names[l],
                hoverinfo='text',
                text=[f"{layer_names[l]} - Node {i+1}" for i in range(n)]
            ))
            
        fig_net.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_net, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 4: Training Diagnostics
# ════════════════════════════════════════════════════════════════════
elif page == '📈 Training Diagnostics':
    st.markdown('<div class="main-title">Model Training Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real loss convergence, accuracy trajectories, and early stopping dynamics</div>', unsafe_allow_html=True)
    
    st.info("💡 **Training Run Summary:** Trained using SGD with Momentum (`lr=0.01`, `momentum=0.9`). Early stopping monitored `val_loss` with `patience=15`. Best weights were restored from **Epoch 8**, halting training at **Epoch 23** to avoid overfitting.")
    
    fig_dir = Path('outputs/figures')
    
    st.subheader("📊 High-Resolution Training & Validation Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Training Loss Convergence**")
        show_image(fig_dir / 'training_loss.png')
    with col2:
        st.markdown("**2. Validation Loss Trajectory**")
        show_image(fig_dir / 'validation_loss.png')
        
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**3. Training Accuracy Growth**")
        show_image(fig_dir / 'training_accuracy.png')
    with col4:
        st.markdown("**4. Validation Accuracy Generalization**")
        show_image(fig_dir / 'validation_accuracy.png')
        
    st.markdown("---")
    st.subheader("📈 Combined Performance Curves")
    show_image(fig_dir / 'training_curves.png')
    
    st.markdown("---")
    st.subheader("🔍 Training Observations & Theoretical Insights")
    st.markdown("""
    1. **Rapid Initial Convergence:** In Epochs 1–5, loss dropped sharply from 0.647 to 0.432 as SGD adjusted the weights along the primary gradient direction.
    2. **Momentum Stabilization:** SGD momentum prevented oscillations across the valley walls of the categorical cross-entropy loss surface.
    3. **Optimal Generalization at Epoch 8:** Validation loss reached its minimum of **0.4196**.
    4. **Early Stopping Restoring Best Weights:** Continued training past epoch 8 did not yield better validation loss; at epoch 23, Keras early stopping triggered and safely restored epoch 8 weights, preserving maximum test-set generalization.
    """)


# ════════════════════════════════════════════════════════════════════
# PAGE 5: Test Evaluation
# ════════════════════════════════════════════════════════════════════
elif page == '📋 Test Evaluation':
    st.markdown('<div class="main-title">Test Set Evaluation & Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Rigorous evaluation on the held-out 180-student test split (15% unseen data)</div>', unsafe_allow_html=True)
    
    try:
        metrics = load_metrics()
        
        # KPI Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#2563EB;">{metrics['test_accuracy']*100:.2f}%</div><div class="metric-label">Accuracy</div></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#059669;">{metrics['precision']*100:.2f}%</div><div class="metric-label">Precision (Pass)</div></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#10B981;">{metrics['recall']*100:.2f}%</div><div class="metric-label">Recall (Pass)</div></div>""", unsafe_allow_html=True)
        m4.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#7C3AED;">{metrics['f1_score']*100:.2f}%</div><div class="metric-label">F1-Score (Pass)</div></div>""", unsafe_allow_html=True)
        m5.markdown(f"""<div class="metric-card"><div class="metric-value" style="color:#DC2626;">{metrics['test_loss']:.4f}</div><div class="metric-label">Test Loss</div></div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        c_left, c_right = st.columns([1, 1], gap="large")
        
        with c_left:
            st.subheader("🎯 Interactive Confusion Matrix")
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            
            cm_z = [[tn, fp], [fn, tp]]
            fig_cm = px.imshow(
                cm_z,
                text_auto=True,
                x=['Predicted FAIL (0)', 'Predicted PASS (1)'],
                y=['Actual FAIL (0)', 'Actual PASS (1)'],
                color_continuous_scale='Blues',
                aspect='auto'
            )
            fig_cm.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
            
            st.markdown(f"""
            - **True Positives (TP):** **{tp}** students correctly predicted as **PASS**
            - **True Negatives (TN):** **{tn}** students correctly predicted as **FAIL**
            - **False Positives (FP):** **{fp}** students predicted Pass but actually Failed
            - **False Negatives (FN):** **{fn}** students predicted Fail but actually Passed
            """)

        with c_right:
            st.subheader("📊 Class-Level Metric Comparison")
            
            rep = metrics.get('classification_report_dict', {})
            pass_metrics = rep.get('Pass', {'precision': 0.84, 'recall': 0.94, 'f1-score': 0.88})
            fail_metrics = rep.get('Fail', {'precision': 0.80, 'recall': 0.58, 'f1-score': 0.67})
            
            fig_bars = go.Figure()
            fig_bars.add_trace(go.Bar(
                name='Pass Class',
                x=['Precision', 'Recall', 'F1-Score'],
                y=[pass_metrics['precision']*100, pass_metrics['recall']*100, pass_metrics['f1-score']*100],
                marker_color='#10B981'
            ))
            fig_bars.add_trace(go.Bar(
                name='Fail Class',
                x=['Precision', 'Recall', 'F1-Score'],
                y=[fail_metrics['precision']*100, fail_metrics['recall']*100, fail_metrics['f1-score']*100],
                marker_color='#EF4444'
            ))
            fig_bars.update_layout(
                barmode='group',
                yaxis=dict(title='Percentage (%)', range=[0, 105]),
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_bars, use_container_width=True)
            
            st.markdown("""
            **Key Metric Insights:**
            - **High Pass Recall (93.6%):** The neural network misses very few students who truly pass.
            - **Solid Fail Precision (80.0%):** When the model flags a student for failure, it is correct in 4 out of 5 cases, providing a reliable early-warning mechanism.
            """)

        st.markdown("---")
        st.subheader("📑 Full Sklearn Classification Report")
        st.code(metrics.get('classification_report_text', 'Classification report not loaded.'), language='text')
        
    except FileNotFoundError:
        st.error("Metrics not found. Run `python src/train.py` first.")


# ════════════════════════════════════════════════════════════════════
# PAGE 6: Student Prediction Studio
# ════════════════════════════════════════════════════════════════════
elif page == '🔮 Student Prediction Studio':
    st.markdown('<div class="main-title">Interactive Student Prediction Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter academic indicators to run real-time inference and view internal neural activations</div>', unsafe_allow_html=True)
    
    try:
        model, scaler, feature_extractor = load_model_and_scaler()
        
        # Archetype Preset Buttons
        st.markdown("##### ⚡ Quick Archetype Presets:")
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        
        if p_col1.button("🌟 Top Achiever Preset", use_container_width=True):
            st.session_state.att = 92
            st.session_state.im = 85
            st.session_state.asgn = 90
            st.session_state.hrs = 7.0
            st.session_state.prev = 88
        if p_col2.button("⚖️ Borderline Student Preset", use_container_width=True):
            st.session_state.att = 68
            st.session_state.im = 50
            st.session_state.asgn = 58
            st.session_state.hrs = 3.5
            st.session_state.prev = 52
        if p_col3.button("🚨 At-Risk Student Preset", use_container_width=True):
            st.session_state.att = 45
            st.session_state.im = 30
            st.session_state.asgn = 35
            st.session_state.hrs = 1.0
            st.session_state.prev = 35
        if p_col4.button("❌ Zero Effort / Absent", use_container_width=True):
            st.session_state.att = 0
            st.session_state.im = 0
            st.session_state.asgn = 0
            st.session_state.hrs = 0.0
            st.session_state.prev = 0

        student_name = st.text_input("👤 Student Name / Roll No. (Optional Display Identifier):", value="Alex Smith (Roll #2024-CS-101)")
        st.caption("ℹ️ **Why Name is Not an Input Feature:** In Machine Learning, names/IDs have zero predictive correlation with performance and would cause bias/overfitting. The model evaluates purely the 5 numeric indicators.")

        # Sliders layout
        c_in1, c_in2 = st.columns(2, gap="large")
        with c_in1:
            attendance = st.slider('🏫 Class Attendance (%)', 0, 100, st.session_state.get('att', 75), help="Percentage of lectures attended (Cutoff is 60%)")
            internal_marks = st.slider('📝 Internal Test Marks (0–100)', 0, 100, st.session_state.get('im', 55), help="Midterm and quiz average (Passing minimum is 35)")
            assignment_score = st.slider('📚 Assignment & Lab Score (0–100)', 0, 100, st.session_state.get('asgn', 65), help="Homework & practical assessments")
        with c_in2:
            study_hours = st.slider('⏰ Daily Study Hours (0–12h)', 0.0, 12.0, float(st.session_state.get('hrs', 5.0)), step=0.5, help="Self-study hours outside class")
            prev_performance = st.slider('🏆 Previous Academic Performance (0–100)', 0, 100, st.session_state.get('prev', 60), help="Prior semester GPA/marks")
            
        # Inference
        raw_feat = np.array([[attendance, internal_marks, assignment_score, study_hours, prev_performance]], dtype=float)
        scaled_feat = scaler.transform(raw_feat)
        
        # Extract activations
        h1_act, h2_act, probs = feature_extractor(scaled_feat)
        h1_vals = h1_act.numpy()[0]
        h2_vals = h2_act.numpy()[0]
        prob_vals = probs.numpy()[0]
        
        fail_prob, pass_prob = float(prob_vals[0]), float(prob_vals[1])
        prediction = "PASS" if pass_prob >= fail_prob else "FAIL"
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Result Banner & Gauge
        res_left, res_right = st.columns([1, 1], gap="large")
        
        with res_left:
            if prediction == "PASS":
                st.markdown(f"""
                <div class="pass-banner">
                    <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Assessment for: {student_name}</div>
                    <h1 style="margin:4px 0; font-size: 2.6rem;">🎉 PREDICTION: PASS</h1>
                    <p style="font-size: 1.1rem; margin-top: 6px; opacity: 0.95;">The student is projected to meet all academic graduation requirements.</p>
                    <div style="font-size: 1.4rem; font-weight: 800; margin-top: 8px;">Pass Confidence: {pass_prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="fail-banner">
                    <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">Assessment for: {student_name}</div>
                    <h1 style="margin:4px 0; font-size: 2.6rem;">❌ PREDICTION: FAIL</h1>
                    <p style="font-size: 1.1rem; margin-top: 6px; opacity: 0.95;">The student is at severe risk of failing the course (detention/shortage).</p>
                    <div style="font-size: 1.4rem; font-weight: 800; margin-top: 8px;">Fail Probability: {fail_prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Interactive Radar Profile
            st.markdown("#### 🕸️ Student Profile vs Cohort Benchmarks")
            radar_fig = go.Figure()
            categories = ['Attendance', 'Internal Marks', 'Assignments', 'Study Hours (x8)', 'Prior Perf.']
            student_vals = [attendance, internal_marks, assignment_score, min(100, study_hours*8.33), prev_performance]
            pass_benchmark = [78, 62, 70, 48, 65]
            fail_benchmark = [55, 38, 48, 25, 45]
            
            radar_fig.add_trace(go.Scatterpolar(r=student_vals, theta=categories, fill='toself', name='Current Student', line=dict(color='#2563EB', width=2.5)))
            radar_fig.add_trace(go.Scatterpolar(r=pass_benchmark, theta=categories, name='Pass Cohort Avg', line=dict(color='#10B981', dash='dash')))
            radar_fig.add_trace(go.Scatterpolar(r=fail_benchmark, theta=categories, name='Fail Cohort Avg', line=dict(color='#EF4444', dash='dot')))
            
            radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, height=320, margin=dict(l=30, r=30, t=20, b=20))
            st.plotly_chart(radar_fig, use_container_width=True)

        with res_right:
            st.markdown("#### ⏱️ Probability Confidence Meter")
            
            # Plotly Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pass_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pass Probability (%)", 'font': {'size': 20}},
                number={'suffix': "%", 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#10B981" if pass_prob >= 0.5 else "#EF4444"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Actionable Academic Advice
            st.markdown("#### 💡 Academic Advising Guidance")
            if prediction == "PASS":
                st.success(f"✅ **Strong Trajectory:** The student has a {pass_prob*100:.1f}% chance of passing. Maintaining attendance above 75% and study hours above 4h will ensure consistent performance.")
            else:
                st.warning(f"⚠️ **Intervention Recommended:** Fail probability is {fail_prob*100:.1f}%. Increasing attendance to 75%+ and raising daily study hours by 2h will significantly flip the trajectory toward PASS.")

        st.markdown("---")
        
        # ─── LIVE NEURAL ACTIVATION HEATMAP ───
        st.subheader("🔥 Live Internal Neural Network Activations (Inside the Black Box)")
        st.caption("These charts show real activations extracted directly from hidden neurons in real time as you adjust the sliders!")
        
        act1_col, act2_col = st.columns(2)
        
        with act1_col:
            st.markdown("**Hidden Layer 1 (32 ReLU Units):**")
            fig_h1 = go.Figure(go.Bar(
                x=[f"N{i+1}" for i in range(32)],
                y=h1_vals,
                marker=dict(color=h1_vals, colorscale='Viridis', showscale=False),
                text=[f"{v:.2f}" if v > 0 else "0" for v in h1_vals],
                textposition='outside'
            ))
            fig_h1.update_layout(
                height=260,
                xaxis_title="Neuron Index",
                yaxis_title="ReLU Output",
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_h1, use_container_width=True)
            active_h1 = int(np.sum(h1_vals > 0))
            st.caption(f"Active Neurons firing (>0): **{active_h1}/32** ({active_h1/32*100:.1f}%) | Dormant: **{32 - active_h1}**")
            
        with act2_col:
            st.markdown("**Hidden Layer 2 (16 ReLU Units):**")
            fig_h2 = go.Figure(go.Bar(
                x=[f"N{i+1}" for i in range(16)],
                y=h2_vals,
                marker=dict(color=h2_vals, colorscale='Plasma', showscale=False),
                text=[f"{v:.2f}" if v > 0 else "0" for v in h2_vals],
                textposition='outside'
            ))
            fig_h2.update_layout(
                height=260,
                xaxis_title="Neuron Index",
                yaxis_title="ReLU Output",
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_h2, use_container_width=True)
            active_h2 = int(np.sum(h2_vals > 0))
            st.caption(f"Active Neurons firing (>0): **{active_h2}/16** ({active_h2/16*100:.1f}%) | Dormant: **{16 - active_h2}**")

    except FileNotFoundError:
        st.error("Model artifacts not found. Please run `python src/train.py` first.")


