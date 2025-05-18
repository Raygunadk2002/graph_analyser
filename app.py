from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import shutil
from pathlib import Path
import logging
import requests
import stumpy
from tslearn.metrics import dtw
import io
from typing import Optional, List
import aiohttp
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Structural Movement Graph Analyser")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_movement(series, dates):
    """Analyze movement patterns in the data"""
    # Convert to monthly averages
    monthly = pd.Series(series.values, index=dates).resample('M').mean()
    
    # Calculate summer (Jun-Aug) and winter (Dec-Feb) averages
    summer = monthly[monthly.index.month.isin([6,7,8])].mean()
    winter = monthly[monthly.index.month.isin([12,1,2])].mean()
    
    # Calculate the seasonal amplitude
    amplitude = abs(summer - winter)
    
    # Calculate trend using linear regression
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    trend_line = slope * x + intercept
    
    # Calculate R-squared to determine how well the trend fits
    y_mean = np.mean(series)
    ss_tot = np.sum((series - y_mean) ** 2)
    ss_res = np.sum((series - trend_line) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    # Calculate seasonal strength
    seasonal_strength = amplitude / (np.std(series) if np.std(series) > 0 else 1)
    
    # Determine if the pattern is consistent (summer opening, winter closing)
    is_consistent = (summer > winter) if series.mean() > 0 else (summer < winter)
    
    # Determine if the movement is progressive
    is_progressive = abs(slope) > 0.1 and r_squared > 0.3
    
    # Determine if the movement is primarily seasonal
    is_seasonal = seasonal_strength > 0.3 and amplitude > 0.1
    
    # Convert all values to float and handle NaN/Inf
    def safe_float(value):
        if pd.isna(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    return {
        'has_seasonal': bool(is_seasonal),
        'amplitude': safe_float(amplitude),
        'is_consistent': bool(is_consistent),
        'summer_avg': safe_float(summer),
        'winter_avg': safe_float(winter),
        'is_progressive': bool(is_progressive),
        'slope': safe_float(slope),
        'r_squared': safe_float(r_squared),
        'seasonal_strength': safe_float(seasonal_strength),
        'movement_type': 'Progressive' if is_progressive else 'Seasonal' if is_seasonal else 'Stable'
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify server is working"""
    return {"status": "success", "message": "Server is working!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and return initial analysis"""
    try:
        logger.debug(f"Received file: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}. Please upload a CSV, XLSX, or XLS file.")
        
        # Clean up old files in upload directory
        try:
            for old_file in UPLOAD_DIR.glob('*'):
                if old_file.is_file():
                    try:
                        old_file.unlink()
                        logger.debug(f"Deleted old file: {old_file}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old file {old_file}: {str(e)}")
            logger.debug("Cleaned up old files in upload directory")
        except Exception as e:
            logger.warning(f"Error cleaning up old files: {str(e)}")
        
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        try:
            # Read file content
            content = await file.read()
            # Close the file handle
            await file.close()
            
            # Write content to new file
            with file_path.open("wb") as buffer:
                buffer.write(content)
            logger.debug(f"File saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
        
        # Read the file based on its extension
        try:
            logger.debug(f"Attempting to read file: {file_path}")
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_path.suffix.lower() == '.xls':
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                except Exception as xlrd_error:
                    logger.warning(f"xlrd error: {str(xlrd_error)}, trying openpyxl")
                    df = pd.read_excel(file_path, engine='openpyxl')
            
            logger.debug(f"File read successfully: {df.shape} rows, {df.columns.tolist()}")
            logger.debug(f"DataFrame info:\n{df.info()}")
            logger.debug(f"First few rows:\n{df.head()}")
            
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
        
        # Basic data validation
        if df.empty:
            raise HTTPException(status_code=400, detail="File contains no data")
        
        # Try to identify datetime column
        datetime_col = None
        logger.debug(f"Available columns: {df.columns.tolist()}")
        logger.debug(f"Column types: {df.dtypes}")
        
        # First try to find a column with 'date' or 'time' in its name
        for col in df.columns:
            if any(term in col.lower() for term in ['date', 'time', 'day', 'month', 'year']):
                try:
                    pd.to_datetime(df[col])
                    datetime_col = col
                    logger.debug(f"Found datetime column by name: {col}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to convert column {col} to datetime: {str(e)}")
                    continue
        
        # If no column found by name, try all columns
        if not datetime_col:
            for col in df.columns:
                try:
                    # Try to convert the first non-null value to datetime
                    sample = df[col].dropna().iloc[0]
                    pd.to_datetime(sample)
                    datetime_col = col
                    logger.debug(f"Found datetime column by content: {col}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to convert column {col} to datetime: {str(e)}")
                    continue
        
        if not datetime_col:
            # If still no datetime column found, provide more detailed error
            column_info = "\n".join([f"- {col}: {df[col].dtype}" for col in df.columns])
            raise HTTPException(
                status_code=400, 
                detail=f"No datetime column found. Please ensure your file has a column with dates.\n\nAvailable columns:\n{column_info}"
            )
        
        logger.debug(f"Datetime column identified: {datetime_col}")
        
        # Convert datetime with more flexible parsing
        try:
            df['__time__'] = pd.to_datetime(df[datetime_col], errors='coerce')
            # Check if any dates were successfully parsed
            if df['__time__'].isna().all():
                raise ValueError("No valid dates could be parsed from the datetime column")
            logger.debug(f"Successfully converted dates. Sample dates: {df['__time__'].head().tolist()}")
        except Exception as e:
            logger.error(f"Error converting datetime: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"Error converting datetime column '{datetime_col}': {str(e)}. Please ensure the dates are in a standard format (e.g., YYYY-MM-DD, DD/MM/YYYY)."
            )
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="No numeric columns found. Please ensure your file has at least one column with numeric data.")
        
        logger.debug(f"Numeric columns: {numeric_cols}")
        
        # Analyze each numeric column
        analysis = {}
        try:
            for col in numeric_cols:
                if col != '__time__':
                    logger.debug(f"Analyzing column: {col}")
                    analysis[col] = analyze_movement(df[col], df['__time__'])
            logger.debug(f"Analysis completed: {analysis}")
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
        
        return {
            "status": "success",
            "message": "File processed successfully",
            "data": {
                "columns": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "datetime_column": datetime_col,
                "analysis": analysis
            }
        }
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/analyze")
async def analyze_data(data: dict):
    """Perform detailed analysis on selected columns"""
    try:
        # Implementation for detailed analysis
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
async def generate_report(data: dict):
    """Generate PDF report"""
    try:
        # Implementation for PDF generation
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/geocode")
async def geocode_postcode(postcode: str):
    """Convert UK postcode to latitude and longitude using postcodes.io"""
    try:
        url = f"https://api.postcodes.io/postcodes/{postcode}"
        resp = requests.get(url)
        if resp.status_code != 200:
            return {"error": f"Postcode lookup failed: {resp.text}"}
        data = resp.json()
        if data.get('status') != 200:
            return {"error": f"Postcode lookup failed: {data.get('error', 'Unknown error')}"}
        lat = data['result']['latitude']
        lon = data['result']['longitude']
        return {"latitude": lat, "longitude": lon}
    except Exception as e:
        return {"error": str(e)}

@app.post("/rainfall")
async def get_rainfall(lat: float, lon: float, start_date: str, end_date: str):
    """Fetch daily rainfall from Open-Meteo for given lat/lon and date range"""
    try:
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}&daily=precipitation_sum&timezone=Europe%2FLondon"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {"error": f"Rainfall API failed: {await resp.text()}"}
                data = await resp.json()
                daily_data = data.get('daily', {})
                if not daily_data:
                    return {"error": "No daily data found in response"}
                
                # Extract dates and rainfall values
                dates = daily_data.get('time', [])
                rainfall = daily_data.get('precipitation_sum', [])
                
                # Ensure we have valid data
                if not dates or not rainfall or len(dates) != len(rainfall):
                    return {"error": "Invalid rainfall data format"}
                
                # Convert rainfall values to numbers and handle any null values
                rainfall = [float(val) if val is not None else 0.0 for val in rainfall]
                
                # Return in the format expected by the frontend
                return {
                    "dates": dates,
                    "rainfall": rainfall
                }
    except Exception as e:
        logger.error(f"Error fetching rainfall data: {str(e)}")
        return {"error": str(e)}

class TimeSeriesData(BaseModel):
    data: List[float]

@app.post("/timeseries-analysis")
async def timeseries_analysis(series: TimeSeriesData):
    """Analyse time series data for patterns, motifs, and anomalies."""
    try:
        logger.debug(f"Received time series data with {len(series.data)} points")
        
        # Clean and validate input
        cleaned_data = []
        for point in series.data:
            if point is None or pd.isna(point):
                continue
            try:
                value = float(point)
                if np.isfinite(value):
                    cleaned_data.append(value)
            except (ValueError, TypeError):
                continue
        
        if len(cleaned_data) < 10:
            logger.error(f"Not enough valid data points: {len(cleaned_data)}")
            raise HTTPException(status_code=422, detail="Not enough valid data points for analysis")
        
        logger.debug(f"Cleaned data has {len(cleaned_data)} valid points")
        
        # Convert to numpy array for analysis
        data_array = np.array(cleaned_data)
        
        # Calculate basic statistics
        data_mean = np.mean(data_array)
        data_std = np.std(data_array)
        
        # Normalize the data
        if data_std > 0:
            normalized_data = (data_array - data_mean) / data_std
        else:
            logger.warning("Data has zero standard deviation, using mean subtraction only")
            normalized_data = data_array - data_mean
        
        # Calculate matrix profile for pattern analysis
        window_size = min(50, len(normalized_data) // 4)
        matrix_profile = stumpy.stump(normalized_data, m=window_size)
        
        # Find motifs (repeating patterns)
        motif_indices = stumpy.motifs(normalized_data, matrix_profile, k=3)
        num_motifs = len(motif_indices) if motif_indices is not None else 0
        
        # Find anomalies
        anomaly_indices = stumpy.anomalies(normalized_data, matrix_profile)
        num_anomalies = len(anomaly_indices) if anomaly_indices is not None else 0
        
        # Classify the pattern type
        pattern_type = classify_pattern(normalized_data)
        
        # Calculate seasonal decomposition
        seasonal = seasonal_decompose(data_array, period=30)
        
        # Determine if the pattern is seasonal
        is_seasonal = np.std(seasonal.seasonal) > 0.1 * np.std(data_array)
        
        # Calculate trend
        trend = np.polyfit(range(len(data_array)), data_array, 1)[0]
        is_trending = abs(trend) > 0.1 * np.std(data_array)
        
        # Determine the overall pattern
        if is_seasonal and is_trending:
            pattern = "Seasonal with Trend"
        elif is_seasonal:
            pattern = "Seasonal"
        elif is_trending:
            pattern = "Trending"
        else:
            pattern = "Stable"
        
        return {
            "patterns": pattern,
            "motifs": f"Found {num_motifs} repeating patterns",
            "anomalies": f"Detected {num_anomalies} anomalies",
            "shape_classification": pattern_type,
            "seasonal_analysis": {
                "is_seasonal": bool(is_seasonal),
                "seasonal_strength": float(np.std(seasonal.seasonal) / np.std(data_array)),
                "trend": float(trend),
                "is_trending": bool(is_trending)
            }
        }
    except Exception as e:
        logger.error(f"Error in timeseries analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def classify_pattern(data):
    """Classify the pattern type of the time series data."""
    # Calculate basic statistics
    mean = np.mean(data)
    std = np.std(data)
    
    # Calculate trend
    x = np.arange(len(data))
    slope = np.polyfit(x, data, 1)[0]
    
    # Calculate seasonality
    fft = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    main_freq = freqs[np.argmax(np.abs(fft[1:len(fft)//2])) + 1]
    
    # Determine pattern type
    if abs(slope) > 0.1 * std:
        if slope > 0:
            return "Upward Trend"
        else:
            return "Downward Trend"
    elif abs(main_freq) > 0.1:
        return "Cyclic"
    elif std > 0.5 * np.mean(np.abs(data)):
        return "Volatile"
    else:
        return "Stable"

def seasonal_decompose(data, period=30):
    """Perform seasonal decomposition on the time series data."""
    # Create a pandas Series with a dummy index
    series = pd.Series(data)
    
    # Calculate moving averages
    trend = series.rolling(window=period, center=True).mean()
    
    # Detrend the data
    detrended = series - trend
    
    # Calculate seasonal component
    seasonal = detrended.rolling(window=period, center=True).mean()
    
    # Calculate residual
    residual = detrended - seasonal
    
    # Create a named tuple to mimic statsmodels decomposition
    from collections import namedtuple
    Decomposition = namedtuple('Decomposition', ['trend', 'seasonal', 'resid'])
    
    return Decomposition(trend=trend, seasonal=seasonal, resid=residual)

@app.post("/debug-timeseries")
async def debug_timeseries_analysis(series: list):
    """Debug time series analysis with detailed steps and diagnostics"""
    results = {
        "received_points": len(series) if series else 0,
        "success": False,
        "steps": [],
        "errors": [],
        "shape_distances": {}
    }
    
    try:
        logger.debug(f"Received debug request with {len(series)} points")
        
        # Step 1: Parse input
        results["steps"].append("Validating input data")
        
        # Clean and validate input data
        clean_arr = []
        for point in series:
            # Handle different types of inputs
            if point is None:
                continue
            try:
                # Convert to float if possible
                value = float(point) if point != "" else None
                if value is not None and np.isfinite(value):
                    clean_arr.append(value)
            except (ValueError, TypeError):
                # Skip values that can't be converted to float
                continue
        
        if not clean_arr or len(clean_arr) < 5:
            results["errors"].append(f"Not enough valid data points: {len(clean_arr)}")
            return results
            
        results["processed_points"] = len(clean_arr)
        results["steps"].append(f"Parsed {len(clean_arr)} valid points out of {len(series)} input points")
        
        # Step 2: Calculate data statistics
        results["steps"].append("Calculating data statistics")
        try:
            raw_arr = np.array(clean_arr)
            min_val = np.min(raw_arr)
            max_val = np.max(raw_arr)
            mean_val = np.mean(raw_arr)
            std_val = np.std(raw_arr)
            has_nan = np.any(np.isnan(raw_arr))
            has_inf = np.any(~np.isfinite(raw_arr))
            
            results["data_stats"] = {
                "min": float(min_val),
                "max": float(max_val),
                "mean": float(mean_val),
                "std": float(std_val),
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
                "nan_count": int(np.sum(np.isnan(raw_arr))),
                "inf_count": int(np.sum(~np.isfinite(raw_arr) & ~np.isnan(raw_arr)))
            }
            results["steps"].append("Successfully calculated data statistics")
        except Exception as e:
            results["errors"].append(f"Failed to calculate statistics: {str(e)}")
            return results
        
        # Step 3: Replace NaNs and Infs with mean
        if has_nan or has_inf:
            results["steps"].append("Replacing NaN/Inf values with mean")
            try:
                raw_arr = np.nan_to_num(raw_arr, nan=mean_val, posinf=mean_val, neginf=mean_val)
            except Exception as e:
                results["errors"].append(f"Failed to replace NaN/Inf values: {str(e)}")
                return results
        
        # Step 4: Apply smoothing
        results["steps"].append("Applying smoothing")
        window_size = min(5, len(raw_arr) // 10)
        if window_size < 2:
            window_size = 2
            
        try:
            arr = np.convolve(raw_arr, np.ones(window_size)/window_size, mode='valid')
            results["steps"].append(f"Applied smoothing with window size {window_size}")
            results["processed_points"] = len(arr)
        except Exception as e:
            results["errors"].append(f"Failed to apply smoothing: {str(e)}")
            arr = raw_arr
            
        # Step 5: Normalize
        results["steps"].append("Normalizing data")
        try:
            arr_mean = np.mean(arr)
            arr_std = np.std(arr)
            if arr_std > 0:
                norm_arr = (arr - arr_mean) / arr_std
            else:
                norm_arr = arr - arr_mean
                results["steps"].append("Data has zero standard deviation")
                if arr_std == 0 and np.all(arr == arr[0]):
                    results["errors"].append("All values are identical, cannot perform analysis")
                    return results
                
            # Check normalized data
            results["norm_stats"] = {
                "min": float(np.min(norm_arr)),
                "max": float(np.max(norm_arr)),
                "mean": float(np.mean(norm_arr)),
                "std": float(np.std(norm_arr))
            }
        except Exception as e:
            results["errors"].append(f"Failed to normalize data: {str(e)}")
            return results
            
        # Step 6: Try stumpy
        results["steps"].append("Attempting matrix profile calculation")
        try:
            window_size = max(3, min(30, len(norm_arr) // 10))
            results["steps"].append(f"Using window size {window_size}")
            
            # Check if the window size is valid
            if window_size >= len(norm_arr):
                results["errors"].append(f"Window size {window_size} is too large for data length {len(norm_arr)}")
                return results
                
            # Check for valid input to stumpy
            if np.any(~np.isfinite(norm_arr)):
                fixed_arr = np.nan_to_num(norm_arr, nan=0.0, posinf=0.0, neginf=0.0)
                results["steps"].append("Fixed non-finite values before matrix profile calculation")
                mp = stumpy.stump(fixed_arr, m=window_size)
            else:
                mp = stumpy.stump(norm_arr, m=window_size)
                
            results["steps"].append("Successfully calculated matrix profile")
            
            # Try motif discovery
            try:
                motif_idx = np.argsort(mp[:, 0])[:2].tolist() if len(mp) > 1 else []
                results["steps"].append(f"Found motifs at indices: {motif_idx}")
            except Exception as me:
                results["errors"].append(f"Error finding motifs: {str(me)}")
                motif_idx = []
            
            # Try discord discovery
            try:
                if len(mp) > 1:
                    discord_idx = stumpy.discord(mp[:, 0], k=1)
                    results["steps"].append(f"Found anomalies at indices: {discord_idx}")
                else:
                    discord_idx = []
                    results["steps"].append("Not enough data for anomaly detection")
            except Exception as de:
                results["errors"].append(f"Error finding anomalies: {str(de)}")
                discord_idx = []
                
        except Exception as e:
            results["errors"].append(f"Failed in matrix profile step: {str(e)}")
            return results
        
        # Step 7: Try shape classification
        results["steps"].append("Attempting shape classification")
        try:
            n = len(norm_arr)
            x = np.linspace(0, 2 * np.pi, n)
            shapes = {
                'Seasonal cycle': np.sin(x),
                'Trend up': np.linspace(0, 1, n),
                'Trend down': np.linspace(1, 0, n),
                'Step change': np.concatenate([np.zeros(n//2), np.ones(n-n//2)]),
                'Spike pattern': np.zeros(n)
            }
            shapes['Spike pattern'][n//2] = 1
            
            # Calculate DTW distances
            distances = {}
            for name, shape in shapes.items():
                try:
                    # Normalize shape too
                    shape_mean = np.mean(shape)
                    shape_std = np.std(shape)
                    if shape_std > 0:
                        normalized_shape = (shape - shape_mean) / shape_std
                    else:
                        normalized_shape = shape - shape_mean
                    
                    # Calculate DTW distance
                    distance = dtw(norm_arr, normalized_shape)
                    distances[name] = float(distance)
                except Exception as e:
                    logger.error(f"DTW failed for {name}: {str(e)}")
                    distances[name] = float('inf')
            
            # Find closest shape
            closest_shape = min(distances, key=distances.get)
            results["steps"].append(f"Closest shape: {closest_shape}")
            results["shape_distances"] = distances
            
        except Exception as e:
            results["errors"].append(f"Failed in shape classification: {str(e)}")
            return results
            
        # Success!
        results["success"] = True
        return results
        
    except Exception as e:
        logger.error(f"Unexpected error in debug_timeseries_analysis: {str(e)}", exc_info=True)
        results["errors"].append(f"Unexpected error: {str(e)}")
        return results

@app.post("/analyse")
async def analyse_file_with_mapping(
    mapping: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """Analyse only the mapped columns (time/x/y/z/t, any of x/y/z/t optional) and return results for only those columns."""
    try:
        # Parse the mapping JSON
        mapping_dict = json.loads(mapping)
        logger.debug(f"Received mapping: {mapping_dict}")
        
        # Get the filename from mapping
        filename = mapping_dict.get('filename')
        if not filename:
            raise HTTPException(status_code=400, detail="No filename provided in mapping")
            
        # Read the file
        file_path = UPLOAD_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
            
        # Read the file based on extension
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
            
        logger.debug(f"Read file with columns: {df.columns.tolist()}")
        
        # Get the mapped columns
        time_col = mapping_dict.get('time')
        x_col = mapping_dict.get('x')
        y_col = mapping_dict.get('y')
        z_col = mapping_dict.get('z')
        t_col = mapping_dict.get('t')
        
        # Create mapped columns dictionary
        mapped_cols = {}
        if x_col and x_col in df.columns:
            mapped_cols['X'] = x_col
        if y_col and y_col in df.columns:
            mapped_cols['Y'] = y_col
        if z_col and z_col in df.columns:
            mapped_cols['Z'] = z_col
        if t_col and t_col in df.columns:
            mapped_cols['T'] = t_col
            
        logger.debug(f"Mapped columns: {mapped_cols}")
        
        # Convert time column to datetime
        df['__time__'] = pd.to_datetime(df[time_col])
        logger.debug(f"Converting time column '{time_col}' to datetime")
        logger.debug(f"Successfully converted time column. First few dates: {df['__time__'].head().tolist()}")
        
        # Prepare plot data
        plot_data = {}
        dates_str = df['__time__'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
        
        # Add each mapped column to plot data
        for axis, col in mapped_cols.items():
            logger.debug(f"Preparing plot data for axis {axis} ({col})")
            plot_data[axis] = {
                'time': dates_str,
                'values': df[col].astype(float).tolist()
            }
            
        logger.debug(f"Successfully prepared plot data for {len(plot_data)} axes")
        
        # Perform analysis
        analysis = {}
        for axis, col in mapped_cols.items():
            logger.debug(f"Analyzing column '{col}' for axis {axis}")
            try:
                result = analyze_movement(df[col], df['__time__'])
                analysis[axis] = {k: None if pd.isna(v) else v for k, v in result.items()}
                logger.debug(f"Analysis for {axis} ({col}): {analysis[axis]}")
            except Exception as e:
                logger.error(f"Error analyzing {axis} ({col}): {str(e)}")
                analysis[axis] = {"error": str(e)}
        
        # Return the results
        logger.debug("Analysis complete, returning results")
        return {
            "status": "success",
            "message": "Analysis complete",
            "data": {
                "columns": list(mapped_cols.values()),
                "datetime_column": time_col,
                "analysis": analysis,
                "plot_data": plot_data
            }
        }
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 