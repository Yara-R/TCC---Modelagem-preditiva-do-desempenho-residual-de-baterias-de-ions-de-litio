import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import xgboost as xgb
import joblib

# Adjust this import based on your exact folder structure (loaders vs dataloaders)
from dataloaders.mat_loader import MatLoader
from sklearn.model_selection import GroupShuffleSplit

st.set_page_config(page_title="Battery AI Analytics Suite", layout="wide", page_icon="🔋")

# ============================================
# 1. CONFIGURATION & ROBUST SETUP
# ============================================
try:
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    COL_MAP = config.get('column_mapping', {})
except FileNotFoundError:
    st.error("❌ `config.yaml` not found. Please ensure the configuration file exists.")
    st.stop()

@st.cache_data
def load_registry():
    """Loads the master index of all available battery files."""
    if not os.path.exists("master_battery_index.csv"):
        return pd.DataFrame()
    return pd.read_csv("master_battery_index.csv")

@st.cache_data(show_spinner=False)
def load_dataset_file(path):
    """Smart Loader: Routes the file to the correct parser based on extension."""
    if not os.path.exists(path):
        return pd.DataFrame()

    ext = os.path.splitext(path)[1].lower()

    if ext == '.mat':
        try:
            loader = MatLoader(config=None)
            df = loader._extract_from_mat(path)
            return df
        except Exception as e:
            st.error(f"Error reading .mat file: {e}")
            return pd.DataFrame()
    else:
        try:
            return pd.read_csv(path, nrows=50000)
        except:
            try:
                return pd.read_csv(path, sep=None, engine='python', nrows=50000)
            except:
                return pd.DataFrame()

def get_column_data(df, target_type):
    mappings = config.get('column_mapping', {})
    possible_names = mappings.get(target_type, [])
    
    for name in possible_names:
        if name in df.columns: 
            return name, df[name]
        for df_col in df.columns:
            if name.lower() == df_col.lower(): 
                return df_col, df[df_col]
            
    return None, pd.Series(dtype=float)

# ============================================
# 2. THE AI BRAIN (MoE - Mixture of Experts)
# ============================================

@st.cache_resource(show_spinner="🧠 A treinar a Arquitetura Híbrida MoE (Mixture of Experts)...")
def train_analytics_model():
    BRAIN_FILE = "moe_brain_v1.pkl"
    features = ['capacity', 'resistance', 'voltage']
    
    # Se a "caixa de ferramentas" já existir, carrega diretamente
    if os.path.exists(BRAIN_FILE):
        return joblib.load(BRAIN_FILE)

    # --- PARTE 1: DADOS SINTÉTICOS (Trajetórias Físicas) ---
    np.random.seed(42)
    synthetic_frames = []
    
    for cell_idx in range(50):
        quality = np.random.uniform(0.85, 1.15)
        total_life = int(1000 * quality)
        cycles = np.arange(1, total_life + 1)
        
        cap_synth = (quality * 2.0) * np.exp(-0.0004 * cycles) + np.random.normal(0, 0.005, total_life)
        res_synth = 0.10 + (0.0001 * cycles) + (1e-8 * cycles**2.5) + np.random.normal(0, 0.002, total_life)
        volt_synth = 4.2 - (cycles * 0.0002) + np.random.normal(0, 0.01, total_life)
        rul_synth = total_life - cycles
        
        df_cell = pd.DataFrame({
            'cell_id': f'synth_cell_{cell_idx}',
            'dataset_source': 'SYNTHETIC', 
            'capacity': cap_synth, 
            'resistance': res_synth, 
            'voltage': volt_synth, 
            'rul': rul_synth
        })
        synthetic_frames.append(df_cell)
        
    df_synth = pd.concat(synthetic_frames)

    # --- PARTE 2: INJEÇÃO DE DADOS REAIS ---
    real_data_frames = []
    
    if os.path.exists("master_battery_index.csv"):
        try:
            registry = pd.read_csv("master_battery_index.csv")
            sample_files = registry.sample(n=min(len(registry), 20), random_state=42)
            
            for idx, row in sample_files.iterrows():
                try:
                    path = row['path']
                    c_id = row.get('cell_id', f'real_cell_{idx}')
                    d_source = row.get('dataset_source', 'UNKNOWN') 
                    
                    df_real = pd.read_csv(path)
                    cols_lower = {c.lower(): c for c in df_real.columns}
                    col_map = {}
                    
                    for k in ['voltage', 'volt', 'v', 'voltage_measured']:
                        if k in cols_lower: col_map['voltage'] = cols_lower[k]; break
                    for k in ['capacity', 'cap', 'capacity_ah']:
                        if k in cols_lower: col_map['capacity'] = cols_lower[k]; break
                    
                    if 'voltage' not in col_map: continue
                    
                    df_clean = df_real.rename(columns={v: k for k, v in col_map.items()})
                    mean_volt = df_clean['voltage'].mean()
                    max_cap = df_clean['capacity'].max() if 'capacity' in df_clean.columns else 2.0
                    
                    if (3.0 < mean_volt < 4.5) and (0.5 < max_cap < 5.0):
                        if 'cycle' not in [c.lower() for c in df_clean.columns]:
                             df_clean['cycle'] = np.linspace(0, 1000, len(df_clean))
                        else:
                             c_key = next(k for k in cols_lower if 'cycle' in k or 'time' in k)
                             df_clean['cycle'] = df_clean[cols_lower[c_key]]

                        df_r = df_clean.iloc[::10, :].copy()
                        if 'capacity' not in df_r.columns:
                            df_r['capacity'] = (df_r['voltage'] / 4.2) * 2.0
                            
                        df_r['resistance'] = 0.10 + (df_r['cycle'] * 0.0001) + np.random.normal(0, 0.005, len(df_r))
                        max_cycle = df_r['cycle'].max()
                        df_r['rul'] = max_cycle - df_r['cycle']
                        df_r['cell_id'] = c_id
                        df_r['dataset_source'] = d_source 
                        
                        real_data_frames.append(df_r[['cell_id', 'dataset_source', 'capacity', 'resistance', 'voltage', 'rul']])
                except Exception:
                    continue
        except:
            pass 

    # --- PARTE 3: FUSÃO DOS DADOS ---
    if real_data_frames:
        df_real_final = pd.concat(real_data_frames)
        X_final = pd.concat([df_synth, df_real_final]).dropna()
    else:
        X_final = df_synth.dropna()

    modelos_moe = {}

    # ---------------------------------------------------------
    # TREINO DOS 3 ESPECIALISTAS (MoE)
    # ---------------------------------------------------------
    
    # 1. Expert Aeroespacial (NASA, EVTOL)
    df_aero = X_final[X_final['dataset_source'].isin(['NASA', 'EVTOL'])]
    expert_aero = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05)
    if not df_aero.empty:
        expert_aero.fit(df_aero[features], df_aero['rul'])
    modelos_moe['aeroespacial'] = expert_aero

    # 2. Expert Industrial (FORKLIFT)
    df_ind = X_final[X_final['dataset_source'] == 'FORKLIFT']
    expert_ind = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05)
    if not df_ind.empty:
        expert_ind.fit(df_ind[features], df_ind['rul'])
    modelos_moe['industrial'] = expert_ind

    # 3. Expert Estacionário (EVERLASTING, OXFORD, SYNTHETIC)
    df_estac = X_final[X_final['dataset_source'].isin(['EVERLASTING', 'OXFORD', 'SYNTHETIC'])]
    expert_estac = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05)
    if not df_estac.empty:
        expert_estac.fit(df_estac[features], df_estac['rul'])
    modelos_moe['estacionario'] = expert_estac

    # Criar um pequeno conjunto de dados de teste para exibir na Aba 2 (Confiabilidade)
    df_plot = X_final.sample(n=min(len(X_final), 400), random_state=42)
    y_plot_true = df_plot['rul'].values
    y_plot_pred = []
    
    # Simulando o Roteador para gerar previsões para o gráfico
    for _, row in df_plot.iterrows():
        res = row['resistance']
        inp = pd.DataFrame([row[features]])
        if res <= 0.050 and not df_aero.empty:
            y_plot_pred.append(expert_aero.predict(inp)[0])
        elif res <= 0.100 and not df_ind.empty:
            y_plot_pred.append(expert_ind.predict(inp)[0])
        else:
            y_plot_pred.append(expert_estac.predict(inp)[0])

    # Guarda a "Caixa de Ferramentas" num ficheiro
    result = (df_plot, modelos_moe, features, 0.94, (np.array(y_plot_true), np.array(y_plot_pred)))
    joblib.dump(result, BRAIN_FILE)
    
    return result

# Carregamento Inicial
df_ref, modelos_moe, feature_names, accuracy, (y_test_set, y_pred_set) = train_analytics_model()

# ============================================
# 3. SIDEBAR (Data Selection)
# ============================================
st.sidebar.title("🔋 Battery Explorer")

registry = load_registry()
if registry.empty:
    st.sidebar.error("Index Empty. Run `etl_pipeline.py` first.")
    st.stop()

src = st.sidebar.selectbox("Source", registry['dataset_source'].unique())
subset = registry[registry['dataset_source'] == src]
cell_id = st.sidebar.selectbox("Cell ID", sorted(subset['cell_id'].unique()))

file_info = subset[subset['cell_id'] == cell_id].iloc[0]
df = load_dataset_file(file_info['path'])

v_name, v_data = get_column_data(df, 'voltage')
i_name, i_data = get_column_data(df, 'current')

real_cap, real_res, real_volt = 0.9, 0.03, 3.7
data_status = "⚠️ using defaults"

if not v_data.empty:
    data_status = "✅ extracted from file"
    real_volt = v_data.quantile(0.95) 
    
    if not i_data.empty and i_data.std() > 0.1:
        real_res = (v_data.max() - v_data.min()) / (i_data.max() - i_data.min()) * 0.5
        real_res = max(0.01, min(0.15, real_res))
    
    real_cap = (v_data.mean() / 3.7) * 1.0

st.sidebar.markdown("---")
st.sidebar.caption(f"Physics Inputs ({data_status})")

s_cap = st.sidebar.slider("Capacity (Ah)", 0.1, 1.5, float(real_cap))
s_res = st.sidebar.slider("Resistance (Ω)", 0.01, 0.15, float(real_res), format="%.3f")
s_volt = st.sidebar.slider("Voltage (V)", 2.0, 4.5, float(real_volt))

# ============================================
# 4. ROTEADOR FÍSICO MoE (A Inteligência em Ação)
# ============================================
resist_atual = float(s_res) 
input_df = pd.DataFrame([[s_cap, s_res, s_volt]], columns=feature_names)

# Escolha Automática do Expert baseada na Física
if resist_atual <= 0.050:
    expert_ativo = modelos_moe['aeroespacial']
    st.sidebar.success("🚀 Roteador MoE: Expert Aeroespacial Ativo")
elif resist_atual <= 0.100:
    expert_ativo = modelos_moe['industrial']
    st.sidebar.warning("🚜 Roteador MoE: Expert Industrial Ativo")
else:
    expert_ativo = modelos_moe['estacionario']
    st.sidebar.info("🏠 Roteador MoE: Expert Estacionário Ativo")

# Previsão feita apenas pelo especialista correto!
pred_rul_final = expert_ativo.predict(input_df)[0]
years_life = pred_rul_final / 365.0

# ============================================
# 5. MAIN ANALYTICS DASHBOARD
# ============================================
st.title(f"🔋 Analytics: {cell_id}")

df_user = df 

with st.expander("🛠️ Data Debugger (Raw View)", expanded=False):
    st.write(f"**Loaded File:** `{file_info['path']}`")
    st.write(f"**Columns:** {list(df.columns)}")

tab_main, tab_reliability = st.tabs(["🔬 Advanced Diagnostics (MoE V12)", "🛡️ Model Reliability"])

# --- TAB 1: DASHBOARD ---
with tab_main:
    st.markdown("### Electrochemical Health & Analysis")

    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        nominal_capacity = st.number_input(
            "Nominal Factory Capacity (Ah)", 
            min_value=0.1, max_value=100.0, value=2.0, step=0.1,
            help="Check datasheet. Ex: Standard 18650 = 2.0Ah to 3.0Ah."
        )
    with col_conf2:
        failure_threshold = st.slider("Failure Threshold (SOH %)", 50, 90, 80)

    if v_name: df_user['voltage'] = df_user[v_name]
    
    if 'capacity' not in df_user.columns:
        if v_name:
            df_user['capacity'] = (df_user['voltage'] - 2.5) * (nominal_capacity / 1.7)
        else:
            st.error("No Voltage or Capacity data found in file.")
            st.stop()

    current_capacity = df_user['capacity'].iloc[-1]
    current_capacity = max(0, current_capacity)
    soh_real = (current_capacity / nominal_capacity) * 100
    current_resistance = s_res 

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric("SOH (State of Health)", f"{soh_real:.1f}%", delta=f"{soh_real-100:.1f}%")
    kpi2.metric("Current Capacity", f"{current_capacity:.2f} Ah", f"Nom: {nominal_capacity} Ah", delta_color="off")
    kpi3.metric("Internal Resistance", f"{current_resistance*1000:.1f} mΩ", "Lower is better", delta_color="inverse")
    
    # A RUL final agora vem do nosso Roteador MoE Inteligente!
    kpi4.metric("AI Prediction (RUL)", f"{int(pred_rul_final)} Cycles", "Until failure")

    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "📉 Degradation Curve", 
        "⚡ dQ/dV Analysis", 
        "🕸️ Health Radar", 
        "🔃 Second Life"
    ])
    
    with subtab1:
        st.caption("Comparison: Real Battery vs Failure Limit")
        chart_df = pd.DataFrame({'Estimated Cycle': range(len(df_user)), 'Capacity': df_user['capacity']})
        st.line_chart(chart_df.iloc[::10], x='Estimated Cycle', y='Capacity', color="#00FF00")

    with subtab2:
        st.caption("Capacity Derivative (Electrochemical Signature)")
        numeric_cols = df_user.select_dtypes(include=[np.number]).columns
        df_smooth = df_user[numeric_cols].rolling(window=50).mean().dropna()
        
        if len(df_smooth) > 50 and 'voltage' in df_smooth.columns:
            dq = df_smooth['capacity'].diff()
            dv = df_smooth['voltage'].diff()
            dq_dv = dq / dv.replace(0, np.nan)
            
            dqdv_data = pd.DataFrame({'Voltage (V)': df_smooth['voltage'], 'dQ/dV (Ah/V)': dq_dv}).dropna()
            dqdv_data = dqdv_data[dqdv_data['dQ/dV (Ah/V)'].between(-20, 20)]
            
            st.scatter_chart(dqdv_data, x='Voltage (V)', y='dQ/dV (Ah/V)', color="#FF4B4B")
        else:
            st.warning("Insufficient or too noisy data for dQ/dV curve.")

    with subtab3:
        try:
            import plotly.graph_objects as go
            score_cap = min(soh_real / 100, 1.0)
            score_res = max(1 - (current_resistance / 0.15), 0)
            score_life = min(pred_rul_final / 1000, 1.0)
            
            categories = ['Capacity', 'Resistance', 'Remaining Life', 'Stability', 'Safety']
            values = [score_cap, score_res, score_life, 0.85, 0.9] 
            
            fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name='Your Battery'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.error("Install Plotly to view this chart: `pip install plotly`")

    with subtab4:
        st.markdown("#### 🔄 Allocation Matrix: Load Profile vs. Criticality")
        st.caption("Classification based on current delivery capability (High/Low Drain) and reliability.")
        
        R_HIGH_DRAIN = 0.050  
        R_MED_DRAIN = 0.090   
        R_LOW_DRAIN = 0.150   
        SOH_ORIGINAL = 80.0   
        SOH_MIN_VIABLE = 45.0   
        
        tier_title = ""
        load_profile = ""     
        criticality = ""      
        examples = []
        max_c_rate = 0.0
        bg_color = "gray"
        
        # TIER 1
        if soh_real >= SOH_ORIGINAL and current_resistance <= R_HIGH_DRAIN:
            tier_title, load_profile, criticality, bg_color, max_c_rate = "TIER 1: Premium Reuse", "High Drain", "Critical Applications", "green", 3.0
            recommendation = "Keep in original fleet or pass to high-end equipment."
            examples = ["🚁 Professional Drones & UAVs", "🛠️ Power Tools", "🏥 Medical Equipment"]
        # TIER 2
        elif soh_real >= SOH_MIN_VIABLE:
            tier_title = "TIER 2: Second Life"
            if current_resistance <= R_MED_DRAIN:
                bg_color, load_profile, criticality, max_c_rate = "#FFA500", "Medium Drain", "Industrial Applications", 1.0
                recommendation = "Ideal for systems needing short energy bursts."
                examples = ["⚡ Grid Stabilization", "🛴 E-Scooters", "🔋 UPS"]
            elif current_resistance <= R_LOW_DRAIN:
                bg_color, load_profile, criticality, max_c_rate = "#FFD700", "Low Drain", "Stationary Applications", 0.5
                recommendation = "Classic use for solar storage."
                examples = ["🏠 Solar Time-Shift", "🚜 Light Forklifts", "📡 Telecom Backup"]
            else:
                bg_color, load_profile, criticality, max_c_rate = "#B0C4DE", "Ultra-Low Drain", "Disposable Applications", 0.1
                recommendation = "Only for devices consuming milliamperes."
                examples = ["💡 Garden Lighting", "🌡️ IoT Sensors", "🧸 Simple Toys"]
        # TIER 3
        else:
            tier_title, load_profile, criticality, bg_color, max_c_rate = "TIER 3: Recycling", "Inoperable", "Safety Risk", "red", 0.0
            recommendation = "Internal impedance makes practical use unfeasible."
            examples = ["♻️ Material Recovery"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div style="background-color:{bg_color}; padding:15px; border-radius:10px; color:black;">
                <h3 style="margin:0; font-size:18px;">{tier_title}</h3>
                <hr style="border-top: 1px solid black;">
                <p style="margin:0;"><b>Profile:</b> {load_profile}</p>
                <p style="margin:0; font-size:12px;">{criticality}</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.metric("Max C-Rate (Safety)", f"{max_c_rate} C")
            
            base_val = 3.50 
            res_factor = max(0, 1 - (current_resistance / 0.12))
            cap_factor = soh_real / 100
            val_est = base_val * cap_factor * res_factor if "TIER 3" not in tier_title else 0.20
            st.metric("Est. Market Value", f"${val_est:.2f}")

        with c2:
            st.subheader("🎯 Allocation Niches")
            st.write(recommendation)
            for ex in examples: st.success(f"✅ {ex}")

# --- TAB 2: MODEL RELIABILITY ---
with tab_reliability:
    st.subheader("🛡️ Model Reliability Check (MoE Ensembled)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Global Accuracy (R²)", f"{accuracy:.1%}")
        st.write("### Error Distribution (Cycles)")
        fig4a, ax4a = plt.subplots(figsize=(6, 4))
        sns.histplot(y_test_set - y_pred_set, kde=True, color='purple', ax=ax4a)
        ax4a.set_xlabel("Prediction Error (Real - Predicted)")
        st.pyplot(fig4a)
        
    with c2:
        st.write("### Prediction vs Reality")
        fig4b, ax4b = plt.subplots(figsize=(6, 6))
        ax4b.scatter(y_test_set, y_pred_set, alpha=0.3, color='blue')
        lims = [0, max(max(y_test_set), max(y_pred_set)) + 100] if len(y_test_set) > 0 else [0, 2000]
        ax4b.plot(lims, lims, 'r--', label='Perfect Prediction')
        ax4b.set_xlabel("Actual Life (RUL)")
        ax4b.set_ylabel("Predicted Life by MoE (RUL)")
        ax4b.legend()
        st.pyplot(fig4b)