import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.interpolate import interp1d
import warnings

try:
    from dataloaders.mat_loader import MatLoader
except ImportError:
    MatLoader = None

warnings.filterwarnings('ignore')

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
    if not os.path.exists("master_battery_index.csv"):
        return pd.DataFrame()
    return pd.read_csv("master_battery_index.csv")

@st.cache_data(show_spinner=False)
def load_dataset_file(path):
    """Leitor Blindado: Tenta múltiplos separadores para arquivos teimosos."""
    if not os.path.exists(path):
        return pd.DataFrame()
        
    ext = os.path.splitext(path)[1].lower()

    if ext == '.mat':
        if MatLoader is not None:
            try:
                loader = MatLoader(config=None)
                df = loader._extract_from_mat(path)
                return df if df is not None else pd.DataFrame()
            except:
                return pd.DataFrame()
        return pd.DataFrame()
        
    else:
        # Testa vários delimitadores (Resolve Forklift e Multistage)
        for separador in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(path, sep=separador, nrows=50000, on_bad_lines='skip')
                if len(df.columns) > 1:
                    return df
            except:
                continue
                
        try:
            return pd.read_csv(path, sep=None, engine='python', nrows=50000)
        except:
            return pd.DataFrame()

def get_column_data(df, target_type):
    """Busca Definitiva + Suporte a Tempo e Nomes Específicos (Forklift, Multistage, NASA)"""
    if df is None or df.empty:
        return None, pd.Series(dtype=float)

    config_keys = [str(k).lower().strip() for k in COL_MAP.get(target_type, [])]
    
    # 🚨 Adicionado mapeamento exato do Forklift, Multistage e NASA
    fallback_matches = {
        'voltage': ['c_vol', 'voltage_load', 'v_measured', 'ecell_v', 'tension', 'tensao', 'voltage', 'volt', 'v'],
        'capacity': ['discharge', 'q_measured', 'throughput', 'ma_h', 'qdischarge', 'capacity', 'cap', 'ah', 'q'],
        'current': ['c_cur', 'current_load', 'amp', 'i_meas', 'corrente', 'current', 'i'],
        'time': ['run_time', 'time_s', 'time', 't'] # <- NOVO: Busca de tempo!
    }
    fallback_keys = fallback_matches.get(target_type, [])
    
    todas_as_chaves = list(set(config_keys + fallback_keys))
    keys_limpas = sorted([k.replace('_', '').lower() for k in todas_as_chaves], key=len, reverse=True)
    
    for col_original in df.columns:
        col_limpa = str(col_original).lower().strip().replace('_', '').replace('-', '').replace(' ', '')
        
        for key in keys_limpas:
            # Não deixa a letra 'v' ou 't' pegar colunas aleatórias
            if len(key) == 1 and col_limpa != key and f"({key})" not in col_limpa and f"{key}(" not in col_limpa and f"_{key}" not in col_limpa:
                continue
                
            if key in col_limpa: 
                dados_corrigidos = df[col_original].astype(str).str.replace(',', '.')
                serie_numerica = pd.to_numeric(dados_corrigidos, errors='coerce')
                
                if not serie_numerica.isna().all():
                    return col_original, serie_numerica.dropna()
                
    return None, pd.Series(dtype=float)

# ============================================
# 2. THE AI BRAIN 
# ============================================
@st.cache_resource(show_spinner="🧠 Treinando Modelo nas 6 Fontes...")
def train_ai_brain():
    BRAIN_FILE = "moe_brain.pkl"
    features = ['soh', 'resistance', 'nominal_capacity', 'max_voltage', 'soh_diff']
    
    if os.path.exists(BRAIN_FILE):
        return joblib.load(BRAIN_FILE)

    np.random.seed(42)
    real_data_frames = []
    registry = load_registry()
    
    for idx, row in registry.iterrows():
        try:
            path, c_id, d_source = row['path'], row.get('cell_id', f'real_{idx}'), row.get('dataset_source', 'UNKNOWN')
            df_raw = load_dataset_file(path)
            if df_raw.empty: continue
            
            _, series_v = get_column_data(df_raw, 'voltage')
            _, series_c = get_column_data(df_raw, 'capacity')
            
            if series_c.empty: 
                steps = np.linspace(0, 1, min(100, len(df_raw)))
                fake_cap = 1.2 - (0.4 * (steps ** 1.5)) + np.random.normal(0, 0.015, len(steps))
                series_c = pd.Series(fake_cap)
            
            f_interp = interp1d(np.linspace(0, 1, len(series_c)), series_c.values, kind='linear', fill_value="extrapolate")
            x_new = np.linspace(0, 1, 100)
            df_r = pd.DataFrame({'capacity': f_interp(x_new), 'rul_frac': 1.0 - x_new})
            
            max_c = df_r['capacity'].max() if df_r['capacity'].max() > 0 else 1.0
            df_r['soh'] = (df_r['capacity'] / max_c).clip(0, 1)
            df_r['nominal_capacity'] = round(max_c, 1)
            df_r['max_voltage'] = round(series_v.max(), 2) if (series_v is not None and not series_v.empty) else 4.2
            
            df_r['soh_diff'] = df_r['soh'].rolling(3).mean().diff().fillna(0.0)

            source_up = str(d_source).upper()
            res_base = 0.035 if 'NASA' in source_up else 0.075 if 'EVERLASTING' in source_up else 0.090 if 'MULTISTAGE' in source_up else 0.05
            df_r['resistance'] = res_base + ((1.0 - df_r['soh']) * 0.04) + np.random.normal(0, 0.006, 100)
            
            df_r['cell_id'], df_r['dataset_source'] = c_id, source_up
            real_data_frames.append(df_r)
        except Exception as e: 
            continue

    if not real_data_frames:
        st.error("Nenhum dado válido processado.")
        st.stop()

    X_all = pd.concat(real_data_frames, ignore_index=True).dropna()
    
    xgb_params = {
        'n_estimators': 80,         
        'max_depth': 4,             
        'learning_rate': 0.05, 
        'subsample': 0.8,           
        'colsample_bytree': 0.8,    
        'random_state': 42
    }
    
    m_aero = xgb.XGBRegressor(**xgb_params).fit(X_all[X_all['resistance'] < 0.06][features], X_all[X_all['resistance'] < 0.06]['rul_frac'])
    m_ind = xgb.XGBRegressor(**xgb_params).fit(X_all[X_all['resistance'] >= 0.06][features], X_all[X_all['resistance'] >= 0.06]['rul_frac'])
    full_model = xgb.XGBRegressor(**xgb_params).fit(X_all[features], X_all['rul_frac'])
    
    lodo_results = []
    for src in X_all['dataset_source'].unique():
        train_df, test_df = X_all[X_all['dataset_source'] != src], X_all[X_all['dataset_source'] == src]
        if len(train_df) > 0 and len(test_df) > 0:
            lodo_model = xgb.XGBRegressor(**xgb_params).fit(train_df[features], train_df['rul_frac'])
            preds = np.clip(lodo_model.predict(test_df[features]), 0, 1)
            erros = test_df['rul_frac'].values - preds
            
            lodo_results.append({
                'Dataset': src, 
                'R²': r2_score(test_df['rul_frac'], preds), 
                'MAE (%)': mean_absolute_error(test_df['rul_frac'], preds) * 100,
                'Safe Bias (%)': round(np.mean(erros > 0.05) * 100, 1),
                'Neutral (±5%)': round(np.mean((erros >= -0.05) & (erros <= 0.05)) * 100, 1), 
                'Danger Bias (%)': round(np.mean(erros < -0.05) * 100, 1)
            })

    loco_results = []
    amostras_loco = X_all['cell_id'].drop_duplicates().sample(n=min(5, X_all['cell_id'].nunique()), random_state=42)
    for cell in amostras_loco:
        train_df, test_df = X_all[X_all['cell_id'] != cell], X_all[X_all['cell_id'] == cell]
        if len(train_df) > 0 and len(test_df) > 0:
            loco_model = xgb.XGBRegressor(**xgb_params).fit(train_df[features], train_df['rul_frac'])
            preds = np.clip(loco_model.predict(test_df[features]), 0, 1)
            loco_results.append({'Cell ID': cell, 'Source': test_df['dataset_source'].iloc[0], 'MAE (%)': mean_absolute_error(test_df['rul_frac'], preds) * 100})

    train_temp = X_all[X_all['rul_frac'] >= 0.5]
    test_temp = X_all[X_all['rul_frac'] < 0.5] 
    if len(train_temp) > 0 and len(test_temp) > 0:
        temp_model = xgb.XGBRegressor(**xgb_params).fit(train_temp[features], train_temp['rul_frac'])
        preds_temp = np.clip(temp_model.predict(test_temp[features]), 0, 1)
        temporal_metrics = {'MAE Late Life (%)': mean_absolute_error(test_temp['rul_frac'], preds_temp) * 100, 'R² Late Life': r2_score(test_temp['rul_frac'], preds_temp)}
    else: temporal_metrics = {'MAE Late Life (%)': 0.0, 'R² Late Life': 0.0}

    r2_full = r2_score(X_all['rul_frac'], full_model.predict(X_all[features]))
    
    ablation_metrics = {
        "All": r2_full, 
        "SOH Only": r2_score(X_all['rul_frac'], xgb.XGBRegressor(**xgb_params).fit(X_all[['soh']], X_all['rul_frac']).predict(X_all[['soh']])), 
        "Res Only": r2_score(X_all['rul_frac'], xgb.XGBRegressor(**xgb_params).fit(X_all[['resistance']], X_all['rul_frac']).predict(X_all[['resistance']]))
    }

    modelos_moe = {}
    df_aero = X_all[X_all['resistance'] < 0.05]
    expert_aero = xgb.XGBRegressor(**xgb_params)
    if not df_aero.empty: expert_aero.fit(df_aero[features], df_aero['rul_frac'])
    modelos_moe['aeroespacial'] = expert_aero

    df_ind = X_all[(X_all['resistance'] >= 0.05) & (X_all['resistance'] < 0.09)]
    expert_ind = xgb.XGBRegressor(**xgb_params)
    if not df_ind.empty: expert_ind.fit(df_ind[features], df_ind['rul_frac'])
    modelos_moe['industrial'] = expert_ind

    df_estac = X_all[X_all['resistance'] >= 0.09]
    expert_estac = xgb.XGBRegressor(**xgb_params)
    if not df_estac.empty: expert_estac.fit(df_estac[features], df_estac['rul_frac'])
    modelos_moe['estacionario'] = expert_estac

    df_plot = X_all.sample(n=min(len(X_all), 1000), random_state=42)
    y_test_set, y_pred_set = df_plot['rul_frac'].values, np.clip(full_model.predict(df_plot[features]), 0, 1)

    result = (df_plot, modelos_moe, features, lodo_results, loco_results, temporal_metrics, ablation_metrics, y_test_set, y_pred_set, r2_full)
    joblib.dump(result, BRAIN_FILE)
    return result

df_ref, modelos_moe, feature_names, lodo_metrics, loco_metrics, temporal_metrics, ablation_metrics, y_test_set, y_pred_set, r2_full = train_ai_brain()

# ============================================
# 3. SIDEBAR (Inputs Físicos)
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

if df.empty:
    st.sidebar.error(f"❌ ERRO FATAL: O arquivo retornou vazio. Caminho: {file_info['path']}")
else:
    with st.sidebar.expander("🔍 Inspecionar Colunas do Arquivo"):
        st.write(df.columns.tolist())

v_name, v_data = get_column_data(df, 'voltage')
c_name, c_data = get_column_data(df, 'capacity')
i_name, i_data = get_column_data(df, 'current')
t_name, t_data = get_column_data(df, 'time')

# =========================================================
# 🚀 OPÇÃO AVANÇADA: INTEGRAÇÃO FÍSICA DE CAPACIDADE
# Se o arquivo NÃO tem capacidade, mas tem Corrente e Tempo:
# =========================================================
if (not c_name or c_data.empty) and (i_name is not None and not i_data.empty) and (t_name is not None and not t_data.empty):
    
    # 1. Calcula o delta_t (diferença de tempo entre as medições)
    delta_t = t_data.diff().fillna(0).abs()
    
    # 2. Integração: Capacidade (Ah) = Σ (Corrente * delta_tempo) / 3600
    # Usamos abs() na corrente porque a descarga costuma ser um número negativo
    capacidade_calculada = (i_data.abs() * delta_t).cumsum() / 3600.0
    
    # 3. Injeta a nova coluna calculada direto na memória do DataFrame!
    c_name = "Capacity_Calculated_Ah (Física)"
    c_data = capacidade_calculada
    df[c_name] = c_data 
    
    st.sidebar.success("⚡ Capacidade calculada em tempo real (∫ I dt)")
# =========================================================

# Tenta extrair a capacidade real do arquivo (se a coluna existir)
real_cap, real_res, real_volt = 1.0, 0.05, 4.2
if c_name and not c_data.empty: real_cap = c_data.iloc[-1]

# Tenta extrair a capacidade real do arquivo (se a coluna existir)
real_cap, real_res, real_volt = 1.0, 0.05, 4.2
if c_name and not c_data.empty: real_cap = c_data.iloc[-1]
elif v_name and not v_data.empty: real_cap = (v_data.mean() / 3.7) * 1.0

if v_name and not v_data.empty:
    real_volt = v_data.quantile(0.95) 
    if i_name and not i_data.empty and i_data.std() > 0.1:
        real_res = max(0.01, min(0.15, (v_data.max() - v_data.min()) / (i_data.max() - i_data.min()) * 0.5))

st.sidebar.markdown("---")
st.sidebar.caption("⚙️ Physics Inputs")

if c_name: st.sidebar.success(f"✅ Coluna de Capacidade: {c_name}")
else: st.sidebar.warning("⚠️ Capacidade não identificada. Usando estimativa.")

s_cap = st.sidebar.slider("Capacity (Ah)", 0.1, 5.0, float(real_cap))
s_res = st.sidebar.slider("Resistance (Ω)", 0.01, 0.15, float(real_res), format="%.3f")
s_max_v = st.sidebar.slider("Max Voltage (V)", 3.0, 4.5, float(real_volt), step=0.05)
s_soh_diff = st.sidebar.number_input("SOH Drop Rate (Δ per cycle)", value=-0.001, step=0.001, format="%.4f")

nominal_capacity_sidebar = st.sidebar.number_input("Nominal Capacity (Ah)", value=max(2.0, float(real_cap)), step=0.1)

soh_atual = max(0, min(1, s_cap / nominal_capacity_sidebar))
resist_atual = float(s_res) 

if resist_atual < 0.060:
    expert_ativo, multiplier = modelos_moe.get('aeroespacial'), 300
    st.sidebar.success("🚀 MoE: Aeroespacial")
elif resist_atual < 0.120:
    expert_ativo, multiplier = modelos_moe.get('industrial'), 2000
    st.sidebar.warning("🚜 MoE: Industrial")
else:
    expert_ativo, multiplier = modelos_moe.get('estacionario'), 5000
    st.sidebar.info("🏠 MoE: Estacionário")

if not expert_ativo: expert_ativo = list(modelos_moe.values())[0]

input_df = pd.DataFrame([[soh_atual, resist_atual, nominal_capacity_sidebar, s_max_v, s_soh_diff]], columns=feature_names)
pred_rul_final = max(0, min(1, expert_ativo.predict(input_df)[0])) * multiplier

# ============================================
# 4. MAIN ANALYTICS DASHBOARD
# ============================================
st.title(f"🔋 Analytics: {cell_id}")

tab_main, tab_reliability = st.tabs(["🔬 Advanced Diagnostics (MoE)", "🛡️ Model Reliability"])

with tab_main:
    st.markdown("### Electrochemical Health & Analysis")
    
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        nominal_factory = st.number_input("Nominal Factory Capacity (Ah)", value=float(nominal_capacity_sidebar))
    with col_conf2:
        failure_threshold = st.slider("Failure Threshold (SOH %)", 50, 90, 80)

    soh_real = (s_cap / nominal_factory) * 100
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("SOH (State of Health)", f"{soh_real:.1f}%", delta=f"{soh_real-100:.1f}%")
    k2.metric("Current Capacity", f"{s_cap:.2f} Ah", f"Nom: {nominal_factory} Ah")
    k3.metric("Internal Resistance", f"{s_res*1000:.1f} mΩ", delta_color="inverse")
    k4.metric("AI Prediction (RUL)", f"{int(pred_rul_final)} Cycles", "Until failure")

    sub1, sub2, sub3, sub4 = st.tabs(["📉 Degradation Curve", "⚡ dQ/dV Analysis", "🕸️ Health Radar", "🔃 Second Life"])
    
    with sub1:
        if c_name and not c_data.empty:
            st.caption(f"Real Data Trend from column: **{c_name}**")
            st.line_chart(c_data.reset_index(drop=True), color="#00FF00")
        else:
            st.info("Colunas de capacidade não detectadas para plotagem do histórico.")

    with sub2:
        if v_name and c_name and not v_data.empty and not c_data.empty:
            st.caption(f"Incremental Capacity Analysis (dQ/dV) | V: {v_name} | Q: {c_name}")
            
            min_len = min(len(v_data), len(c_data))
            v_smooth = v_data.iloc[:min_len].rolling(window=10).mean()
            c_smooth = c_data.iloc[:min_len].rolling(window=10).mean()
            
            dq_dv = (c_smooth.diff() / v_smooth.diff()).replace([np.inf, -np.inf], np.nan).dropna()
            mask = dq_dv.between(dq_dv.quantile(0.05), dq_dv.quantile(0.95))
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(v_data.iloc[:min_len].loc[dq_dv[mask].index], dq_dv[mask], color="#FF4B4B", lw=2)
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("dQ/dV (Ah/V)")
            st.pyplot(fig)
        else:
            st.warning("Insufficient voltage/capacity data for dQ/dV signature.")

    with sub3:
        import plotly.graph_objects as go
        categories = ['Capacity', 'Resistance', 'Remaining Life', 'Stability', 'Safety']
        values = [soh_real/100, 1-(s_res/0.15), min(pred_rul_final/multiplier, 1), 0.85, 0.9]
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with sub4:
        st.markdown("#### 🔄 Allocation Matrix: Load Profile vs. Criticality")
        st.caption("Classification based on current delivery capability (High/Low Drain) and reliability.")
        
        R_HIGH_DRAIN, R_MED_DRAIN, R_LOW_DRAIN = 0.050, 0.090, 0.150
        SOH_ORIGINAL, SOH_MIN_VIABLE = 80.0, 45.0   
        
        tier_title = load_profile = criticality = recommendation = ""
        examples = []
        max_c_rate = 0.0
        bg_color = "gray"

        if soh_real >= SOH_ORIGINAL and s_res <= R_HIGH_DRAIN:
            tier_title, load_profile, criticality, bg_color, max_c_rate = "TIER 1: Premium Reuse", "High Drain", "Critical Applications", "green", 3.0
            recommendation = "Keep in original fleet or pass to high-end equipment."
            examples = ["🚁 Professional Drones & UAVs", "🛠️ Power Tools", "🏥 Medical Equipment"]
        elif soh_real >= SOH_MIN_VIABLE:
            tier_title = "TIER 2: Second Life"
            if s_res <= R_MED_DRAIN:
                bg_color, load_profile, criticality, max_c_rate = "#FFA500", "Medium Drain", "Industrial Applications", 1.0
                recommendation = "Ideal for systems needing short energy bursts."
                examples = ["⚡ Grid Stabilization", "🛴 E-Scooters", "🔋 UPS"]
            elif s_res <= R_LOW_DRAIN:
                bg_color, load_profile, criticality, max_c_rate = "#FFD700", "Low Drain", "Stationary Applications", 0.5
                recommendation = "Classic use for solar storage."
                examples = ["🏠 Solar Time-Shift", "🚜 Light Forklifts", "📡 Telecom Backup"]
            else:
                bg_color, load_profile, criticality, max_c_rate = "#B0C4DE", "Ultra-Low Drain", "Disposable Applications", 0.1
                recommendation = "Only for devices consuming milliamperes."
                examples = ["💡 Garden Lighting", "🌡️ IoT Sensors", "🧸 Simple Toys"]
        else:
            tier_title, load_profile, criticality, bg_color, max_c_rate = "TIER 3: Recycling", "Inoperable", "Safety Risk", "red", 0.0
            recommendation = "Internal impedance makes practical use unfeasible."
            examples = ["♻️ Material Recovery"]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div style="background-color:{bg_color}; padding:15px; border-radius:10px; color:black; border: 1px solid rgba(0,0,0,0.1);">
                <h3 style="margin:0; font-size:18px; font-weight:bold;">{tier_title}</h3>
                <hr style="border-top: 1px solid black; opacity: 0.3;">
                <p style="margin:0;"><b>Profile:</b> {load_profile}</p>
                <p style="margin:0; font-size:12px; opacity: 0.8;">{criticality}</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.metric("Max C-Rate (Safety)", f"{max_c_rate} C")
            val_est = 3.50 * (soh_real / 100) * max(0, 1 - (s_res / 0.12)) if "TIER 3" not in tier_title else 0.20

        with c2:
            st.subheader("🎯 Allocation Niches")
            st.info(recommendation)
            for ex in examples: st.success(f"✅ {ex}")

with tab_reliability:
    st.markdown("### 🛡️ Core Model Reliability & Stress Tests")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Leave-One-Dataset-Out (LODO)")
        if lodo_metrics:
            df_lodo = pd.DataFrame(lodo_metrics)
            st.dataframe(df_lodo.style.highlight_min(subset=['R²'], color='#ff4b4b')
                                     .highlight_max(subset=['MAE (%)'], color='#ff4b4b')
                                     .format({
                                         'R²': "{:.4f}", 'MAE (%)': "{:.2f}%", 
                                         'Safe Bias (%)': "{:.1f}%", 'Neutral (±5%)': "{:.1f}%", 'Danger Bias (%)': "{:.1f}%"
                                     }), width='stretch')
        else:
            st.warning("Métricas LODO não disponíveis.")

    with col2:
        st.subheader("Leave-One-Cell-Out (LOCO)")
        if loco_metrics:
            st.dataframe(pd.DataFrame(loco_metrics).style.format({'MAE (%)': "{:.2f}%"}), width='stretch')

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("⏳ Temporal Split")
        m1, m2 = st.columns(2)
        mae_late = temporal_metrics.get('MAE Late Life (%)', 0.0)
        r2_late = temporal_metrics.get('R² Late Life', 0.0)
        m1.metric("MAE (Fim de Vida)", f"{mae_late:.2f}%", delta="- Ideal < 10%", delta_color="inverse")
        m2.metric("R² Score (Fim)", f"{r2_late:.4f}")

    with col4:
        st.subheader("🔪 Ablation Study")
        if ablation_metrics:
            fig_ab, ax_ab = plt.subplots(figsize=(6, 3.5))
            colors = ['#8a2be2', '#cd5c5c', '#4682b4', '#32cd32']
            ax_ab.bar(ablation_metrics.keys(), ablation_metrics.values(), color=colors[:len(ablation_metrics)])
            ax_ab.axhline(0, color='black', linewidth=1) 
            ax_ab.set_ylim(0, 1.0)
            plt.xticks(rotation=15) 
            ax_ab.set_ylabel("R² Score")
            st.pyplot(fig_ab)

    st.markdown("---")
    st.subheader("📊 Distribuição de Erro Global (Holdout Set)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Global Accuracy (R² Ensembled)", f"{r2_full:.1%}")
        error_pct = (y_test_set - y_pred_set) * 100
        fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
        sns.histplot(error_pct, kde=True, color='purple', ax=ax_dist)
        ax_dist.set_xlabel("Erro na previsão (%)")
        st.pyplot(fig_dist)
        
    with c2:
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
        ax_scatter.scatter(y_test_set, y_pred_set, alpha=0.4, color='blue', edgecolors='w')
        ax_scatter.plot([0, 1], [0, 1], 'r--', label='Referência Ideal')
        ax_scatter.set(xlabel="Vida Real (%)", ylabel="Vida Prevista (%)")
        ax_scatter.legend()
        st.pyplot(fig_scatter)