# Structural Movement Graph Analyser

A web application for analyzing and visualizing structural movement data, including temperature and displacement measurements.

## Features

- Upload and process Excel/CSV files containing structural movement data
- Interactive visualization of movement patterns
- Scatter matrix view to quickly spot linear relationships
- Analysis of seasonal patterns and trends (displacement only)
- Rainfall and temperature cross-correlation analysis (runs automatically when the server starts)
- Export capabilities for reports
- Advanced analysis including STL decomposition, change point detection,
  rolling trend estimation, frequency spectrum and Kalman filtering

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Raygunadk2002/graph_analyser.git
cd graph_analyser
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```
The application will be available at `http://localhost:8000`.
When the server starts, a correlation analysis between movement,
rainfall and temperature data is executed automatically and the
generated plots are saved in the `analysis_outputs` directory.
If you wish to use real monitoring, rainfall and temperature data
place CSV files named `monitoring.csv`, `rainfall.csv` and
`temperature.csv` inside a directory called `data`. These files will
be loaded automatically instead of the example data that is generated
for demonstration purposes.

## Usage

1. Open `http://localhost:8000` and create an account or log in.
2. After logging in you will be redirected to the dashboard at `/dashboard`.
3. Create a project and upload one or more files for analysis.
4. To view rainfall and temperature correlations for an uploaded file, open
   `/correlation?file_id=YOUR_FILE_ID`. The page will automatically run the
   correlation analysis for that file and display the resulting plots.

## Data Format

The application expects data files (Excel or CSV) with the following columns:
- A datetime column (e.g., 'date')
- Movement data columns (X, Y, Z)
- Temperature data column (T)

## Dependencies

- FastAPI
- Pandas
- NumPy
- Plotly
- Uvicorn

## License

MIT License
