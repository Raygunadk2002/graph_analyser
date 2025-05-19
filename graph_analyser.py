import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from fpdf import FPDF
import os
import pyexcel
from PIL import Image
import io

# Set page config at the very start
st.set_page_config(
    page_title="Structural Movement Graph Analyser",
    page_icon="📐",
    layout="wide"
)

# Modern header with colored bar
st.markdown(
    """
    <div style='background-color: #2563eb; padding: 1.5rem 1rem 1rem 1rem; border-radius: 0 0 1rem 1rem; margin-bottom: 2rem;'>
        <div style='display: flex; align-items: center;'>
            <div>
                <h1 style='color: #fff; margin-bottom: 0;'>📐 Structural Movement Graph Analyser</h1>
                <p style='color: #e0e7ef; margin-top: 0.2rem;'>Modern tool for visualizing and reporting structural movement data</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Optionally show the local logo below the header if available
# logo_path = "Moniteye+Logo+Correct+Blue.jpeg"
# if os.path.exists(logo_path):
#     st.image(logo_path, width=120)

def read_file_data(uploaded_file):
    """Read file data using pandas"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            return pd.read_excel(uploaded_file, engine='pyexcel')
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Sidebar with expanders and About section
with st.sidebar:
    with st.expander("📋 Job & Site Info", expanded=True):
        job_number = st.text_input("Job Number")
        client = st.text_input("Client")
        address = st.text_input("Address")
        requested_by = st.text_input("Requested By")
        postcode = st.text_input("Postcode (for rainfall)", placeholder="e.g. SW1A 1AA")
    with st.expander("📂 Data & Options", expanded=True):
        st.markdown("### Upload Data")
        st.markdown("Supported formats: CSV, XLS, XLSX")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xls", "xlsx"],
            help="Upload your data file here"
        )
        
        if uploaded_file is not None:
            try:
                df = read_file_data(uploaded_file)
                if df is not None and not df.empty:
                    st.success(f"Successfully loaded {len(df)} rows of data")
                else:
                    st.error("Could not read the uploaded file")
                    st.stop()
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.stop()
        
        st.markdown("### Analysis Options")
        include_rain = st.checkbox("Include Rainfall Data", value=True)
        include_soil = st.checkbox("Include Soil Moisture Deficit", value=False)
        components = st.multiselect(
            "Show Components",
            ["Original","Thermal","Seasonal","Progressive"],
            default=["Original","Thermal","Seasonal","Progressive"]
        )
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        **Structural Movement Graph Analyser** helps you visualize, decompose, and report on structural sensor data. Upload your data, select options, and view interactive charts and summaries.\
        
        - Built with Streamlit & Plotly\
        - Modern UI and PDF export\
        - Developed by Moniteye
        """)

tabs = st.tabs(["📊 Graph View","📈 Summary","�� PDF Report"])

def get_latlon(postcode):
    """Get latitude and longitude from postcode with better error handling"""
    if not postcode or not postcode.strip():
        return None, None
        
    try:
        # Clean the postcode
        postcode = postcode.strip().replace(" ", "").upper()
        r = requests.get(f"https://api.postcodes.io/postcodes/{postcode}", timeout=5)
        if r.status_code == 200:
            d = r.json().get('result',{})
            return d.get('latitude'), d.get('longitude')
        else:
            st.warning(f"Could not get location for postcode {postcode}. Rainfall data will not be available.")
            return None, None
    except Exception as e:
        st.warning(f"Error getting location data: {str(e)}. Rainfall data will not be available.")
        return None, None

def get_rainfall(lat, lon):
    """Get rainfall data with better error handling"""
    if lat is None or lon is None:
        return None
        
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=Europe%2FLondon"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            df = pd.DataFrame(r.json().get('daily',{}))
            df['time'] = pd.to_datetime(df['time'])
            return df.set_index('time')['precipitation_sum']
        else:
            st.warning("Could not get rainfall data. Continuing without rainfall information.")
            return None
    except Exception as e:
        st.warning(f"Error getting rainfall data: {str(e)}. Continuing without rainfall information.")
        return None

def get_soil_moisture(lat, lon):
    # Placeholder for COSMOS-UK integration
    return None

def smart_datetime(df):
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
            if parsed.notna().sum() > len(df)*0.5:
                return col, parsed
        except:
            continue
    return None, None

def decompose(series, env=None):
    if env is not None:
        mask = series.notna() & env.notna()
        coeff = np.polyfit(env[mask], series[mask], 1)
        thermal = coeff[0]*env + coeff[1]
    else:
        thermal = pd.Series(0, index=series.index)
    residual = series - thermal
    seasonal = residual.rolling(window=30, min_periods=1, center=True).mean()
    progressive = series - thermal - (seasonal - residual.mean())
    return thermal, seasonal, progressive

def classify_strength(v):
    v = abs(v)
    if v < 0.3: return "weak"
    if v < 0.6: return "moderate"
    return "strong"

def analyze_trend(series):
    s = pd.Series(series).dropna().reset_index(drop=True)
    if len(s) < 5: return "none","insufficient"
    slope = np.polyfit(range(len(s)), s, 1)[0]
    return "progressive", classify_strength(slope/s.std())

def analyze_seasonal_pattern(series, dates):
    """Assess if movement moves away from 0 in summer and back towards 0 in winter."""
    # Convert to monthly averages
    monthly = pd.Series(series.values, index=dates).resample('M').mean()

    # Summer: April to September, Winter: October to March
    summer = monthly[monthly.index.month.isin([4,5,6,7,8,9])].mean()
    winter = monthly[monthly.index.month.isin([10,11,12,1,2,3])].mean()

    # Seasonal amplitude
    amplitude = abs(summer - winter)

    # Movement should be larger in summer and near zero in winter
    winter_near_zero = abs(winter) < 0.2
    is_consistent = abs(summer) > abs(winter)
    has_seasonal = amplitude > 0.2 and winter_near_zero and is_consistent

    return {
        'has_seasonal': has_seasonal,
        'amplitude': amplitude,
        'is_consistent': is_consistent,
        'summer_avg': summer,
        'winter_avg': winter
    }

def analyze_movement(series, dates):
    """Comprehensive analysis of movement patterns"""
    # Basic trend analysis
    trend, strength = analyze_trend(series)
    
    # Seasonal analysis
    seasonal = analyze_seasonal_pattern(series, dates)
    
    # Progressive analysis
    slope = np.polyfit(range(len(series)), series, 1)[0]
    is_progressive = abs(slope) > 0.1  # Threshold for progressive movement
    
    # Determine the primary movement type
    if seasonal['has_seasonal'] and seasonal['is_consistent']:
        movement_type = "Seasonal (Clay Shrinkage)"
        explanation = "Movement shows clear seasonal pattern consistent with clay shrinkage (summer opening, winter closing)"
    elif is_progressive:
        movement_type = "Progressive"
        explanation = "Movement shows continuous opening/closing, potentially indicating drainage issues or undermining"
    else:
        movement_type = "Stable"
        explanation = "No significant seasonal or progressive movement detected"
    
    return {
        'type': movement_type,
        'explanation': explanation,
        'trend': trend,
        'strength': strength,
        'seasonal': seasonal,
        'is_progressive': is_progressive,
        'slope': slope
    }

if uploaded_file is not None and 'df' in locals():
    try:
        # Process the data
        dt_col, parsed = smart_datetime(df)
        if dt_col:
            df['__time__'] = parsed
            st.success(f"Automatically identified time column: {dt_col}")
        else:
            dt_col = st.sidebar.selectbox("Select Time Column", df.columns)
            df['__time__'] = pd.to_datetime(df[dt_col], errors='coerce')
        
        df = df.dropna(subset=['__time__']).sort_values('__time__')
        
        available_columns = [c for c in df.columns if c != '__time__']
        sensor_cols = st.sidebar.multiselect(
            "Select Sensor Columns",
            available_columns,
            default=available_columns[:1] if available_columns else None
        )
        
        if not sensor_cols:
            st.error("Please select at least one sensor column to analyze.")
            st.stop()
        
        # Convert sensor data to numeric
        for c in sensor_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Create tabs
        tabs = st.tabs(["📊 Graph View", "📈 Analysis", "🖨 PDF Report"])
        
        ### Graph View
        with tabs[0]:
            st.markdown("#### Sensor Data Visualization")
            fig = go.Figure()
            for c in sensor_cols:
                orig = df[c]
                if "Original" in components:
                    fig.add_trace(go.Scatter(x=df['__time__'], y=orig, name=f"{c} orig"))
            if rain is not None:
                fig.add_trace(go.Bar(x=df['__time__'], y=rain, name='Rainfall', yaxis='y2', opacity=0.3, marker_color='#2563eb'))
            fig.update_layout(
                template="plotly_white",
                xaxis_title="Time", 
                yaxis_title="Sensor",
                yaxis2=dict(overlaying='y', side='right', title='Rainfall'),
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

        ### Summary
        with tabs[1]:
            st.markdown("#### Movement Analysis")
            for c in sensor_cols:
                with st.expander(f"Analysis for {c}", expanded=True):
                    # Perform comprehensive analysis
                    analysis = analyze_movement(df[c], df['__time__'])
                    
                    # Display movement type and explanation
                    st.markdown(f"**Movement Type:** {analysis['type']}")
                    st.markdown(f"**Explanation:** {analysis['explanation']}")
                    
                    # Show decomposition plot
                    fig = go.Figure()
                    orig = df[c]
                    thermal, seasonal, prog = decompose(orig, rain if include_rain else None)
                    
                    fig.add_trace(go.Scatter(x=df['__time__'], y=orig, name="Original", line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=df['__time__'], y=thermal, name="Thermal", line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=df['__time__'], y=seasonal, name="Seasonal", line=dict(color='green')))
                    fig.add_trace(go.Scatter(x=df['__time__'], y=prog, name="Progressive", line=dict(color='purple')))
                    
                    if rain is not None:
                        fig.add_trace(go.Bar(x=df['__time__'], y=rain, name='Rainfall', yaxis='y2', opacity=0.3, marker_color='#2563eb'))
                    
                    fig.update_layout(
                        template="plotly_white",
                        title="Movement Decomposition",
                        xaxis_title="Time",
                        yaxis_title="Movement",
                        yaxis2=dict(overlaying='y', side='right', title='Rainfall'),
                        height=400,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Trend", analysis['trend'], analysis['strength'])
                        if analysis['seasonal']['has_seasonal']:
                            st.metric("Seasonal Amplitude", f"{analysis['seasonal']['amplitude']:.2f}mm")
                    with col2:
                        if rain is not None:
                            corr_r = df[c].corr(rain)
                            st.metric("Rainfall Correlation", f"{corr_r:.2f}")
                        if soil is not None:
                            corr_s = df[c].corr(soil)
                            st.metric("Soil Moisture Correlation", f"{corr_s:.2f}")

        ### PDF Report
        with tabs[2]:
            if st.button("Download PDF Report"):
                try:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Header
                    pdf.cell(200,10,txt="Structural Movement Analysis Report",ln=1)
                    pdf.cell(200,10,txt=f"Job Number: {job_number}",ln=1)
                    pdf.cell(200,10,txt=f"Client: {client}",ln=1)
                    pdf.cell(200,10,txt=f"Address: {address}",ln=1)
                    pdf.cell(200,10,txt=f"Requested By: {requested_by}",ln=1)
                    pdf.ln(5)
                    
                    # Analysis for each sensor
                    for c in sensor_cols:
                        analysis = analyze_movement(df[c], df['__time__'])
                        pdf.cell(200,10,txt=f"\nSensor: {c}",ln=1)
                        pdf.cell(200,10,txt=f"Movement Type: {analysis['type']}",ln=1)
                        pdf.cell(200,10,txt=f"Explanation: {analysis['explanation']}",ln=1)
                        
                        if analysis['seasonal']['has_seasonal']:
                            pdf.cell(200,10,txt=f"Seasonal Amplitude: {analysis['seasonal']['amplitude']:.2f}mm",ln=1)
                            pdf.cell(200,10,txt="This pattern is typical of clay shrinkage and swelling.",ln=1)
                            pdf.cell(200,10,txt="Summer (Jun-Aug) average: {:.2f}mm".format(analysis['seasonal']['summer_avg']),ln=1)
                            pdf.cell(200,10,txt="Winter (Dec-Feb) average: {:.2f}mm".format(analysis['seasonal']['winter_avg']),ln=1)
                        
                        if analysis['is_progressive']:
                            pdf.cell(200,10,txt="Progressive movement detected - may indicate:",ln=1)
                            pdf.cell(200,10,txt="- Drainage issues",ln=1)
                            pdf.cell(200,10,txt="- Property undermining",ln=1)
                            pdf.cell(200,10,txt="- Structural concerns",ln=1)
                            pdf.cell(200,10,txt=f"Movement rate: {analysis['slope']:.2f}mm per time period",ln=1)
                        
                        if rain is not None:
                            corr_r = df[c].corr(rain)
                            pdf.cell(200,10,txt=f"Rainfall correlation: {corr_r:.2f}",ln=1)
                        
                        pdf.ln(5)
                    
                    # Add summary section
                    pdf.add_page()
                    pdf.cell(200,10,txt="Summary of Findings",ln=1)
                    pdf.ln(5)
                    
                    seasonal_sensors = [c for c in sensor_cols if analyze_movement(df[c], df['__time__'])['seasonal']['has_seasonal']]
                    progressive_sensors = [c for c in sensor_cols if analyze_movement(df[c], df['__time__'])['is_progressive']]
                    
                    if seasonal_sensors:
                        pdf.cell(200,10,txt="Sensors showing seasonal movement (clay shrinkage):",ln=1)
                        for c in seasonal_sensors:
                            pdf.cell(200,10,txt=f"- {c}",ln=1)
                        pdf.ln(5)
                    
                    if progressive_sensors:
                        pdf.cell(200,10,txt="Sensors showing progressive movement:",ln=1)
                        for c in progressive_sensors:
                            pdf.cell(200,10,txt=f"- {c}",ln=1)
                        pdf.ln(5)
                    
                    # Recommendations section
                    pdf.add_page()
                    pdf.cell(200,10,txt="Recommendations",ln=1)
                    pdf.ln(5)
                    
                    if seasonal_sensors:
                        pdf.cell(200,10,txt="For seasonal movement:",ln=1)
                        pdf.cell(200,10,txt="- Consider clay soil management",ln=1)
                        pdf.cell(200,10,txt="- Review drainage systems",ln=1)
                        pdf.cell(200,10,txt="- Monitor tree root systems",ln=1)
                        pdf.ln(5)
                    
                    if progressive_sensors:
                        pdf.cell(200,10,txt="For progressive movement:",ln=1)
                        pdf.cell(200,10,txt="- Investigate drainage systems",ln=1)
                        pdf.cell(200,10,txt="- Check for water leaks",ln=1)
                        pdf.cell(200,10,txt="- Consider structural assessment",ln=1)
                        pdf.ln(5)
                    
                    out = "/tmp/report_v11.pdf"
                    pdf.output(out)
                    with open(out, "rb") as f:
                        st.download_button("Download Report", f, file_name="report_v11.pdf")
                except Exception as e:
                    st.error(f"Error generating PDF report: {str(e)}")
                    st.info("Please try again or contact support if the issue persists.")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()
