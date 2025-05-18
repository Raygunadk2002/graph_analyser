import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    upload_time TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL
                )
            ''')
            
            # Create analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    time_column TEXT NOT NULL,
                    x_column TEXT,
                    y_column TEXT,
                    z_column TEXT,
                    t_column TEXT,
                    analysis_time TIMESTAMP NOT NULL,
                    FOREIGN KEY (file_id) REFERENCES files (id)
                )
            ''')
            
            # Create results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    axis TEXT NOT NULL,
                    has_seasonal BOOLEAN,
                    amplitude REAL,
                    is_consistent BOOLEAN,
                    summer_avg REAL,
                    winter_avg REAL,
                    is_progressive BOOLEAN,
                    slope REAL,
                    r_squared REAL,
                    seasonal_strength REAL,
                    movement_type TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analysis (id)
                )
            ''')
            
            self.conn.commit()
            logger.debug("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def store_file(self, filename: str, file_path: str) -> int:
        """Store file information in the database."""
        try:
            # Convert to absolute path
            abs_path = str(Path(file_path).resolve())
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO files (filename, upload_time, file_path, status) VALUES (?, ?, ?, ?)",
                (filename, datetime.now(), abs_path, "uploaded")
            )
            self.conn.commit()
            file_id = cursor.lastrowid
            logger.debug(f"Stored file {filename} with ID {file_id} at path {abs_path}")
            return file_id
        except Exception as e:
            logger.error(f"Error storing file: {str(e)}")
            raise
    
    def store_analysis(self, file_id: int, mapping: Dict[str, str]) -> int:
        """Store analysis configuration in the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO analysis 
                (file_id, time_column, x_column, y_column, z_column, t_column, analysis_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    mapping.get('time'),
                    mapping.get('x'),
                    mapping.get('y'),
                    mapping.get('z'),
                    mapping.get('t'),
                    datetime.now()
                )
            )
            self.conn.commit()
            analysis_id = cursor.lastrowid
            logger.debug(f"Stored analysis configuration with ID {analysis_id}")
            return analysis_id
        except Exception as e:
            logger.error(f"Error storing analysis: {str(e)}")
            raise
    
    def store_results(self, analysis_id: int, axis: str, results: Dict[str, Any]):
        """Store analysis results in the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO results 
                (analysis_id, axis, has_seasonal, amplitude, is_consistent, 
                summer_avg, winter_avg, is_progressive, slope, r_squared, 
                seasonal_strength, movement_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    axis,
                    results.get('has_seasonal'),
                    results.get('amplitude'),
                    results.get('is_consistent'),
                    results.get('summer_avg'),
                    results.get('winter_avg'),
                    results.get('is_progressive'),
                    results.get('slope'),
                    results.get('r_squared'),
                    results.get('seasonal_strength'),
                    results.get('movement_type')
                )
            )
            self.conn.commit()
            logger.debug(f"Stored results for analysis {analysis_id}, axis {axis}")
        except Exception as e:
            logger.error(f"Error storing results: {str(e)}")
            raise
    
    def get_file_data(self, file_id: int) -> Optional[pd.DataFrame]:
        """Retrieve file data from the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM files WHERE id = ?", (file_id,))
            result = cursor.fetchone()
            
            if result is None:
                logger.error(f"No file found with ID {file_id}")
                return None
            
            file_path = result[0]
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found at path: {file_path}")
                return None
            
            # Read the file based on its extension
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                    except Exception as openpyxl_error:
                        logger.warning(f"openpyxl error: {str(openpyxl_error)}, trying xlrd")
                        df = pd.read_excel(file_path, engine='xlrd')
                else:
                    logger.error(f"Unsupported file type: {file_path}")
                    return None
                
                logger.debug(f"Retrieved data for file {file_id}: {df.shape} rows")
                return df
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error retrieving file data: {str(e)}")
            raise
    
    def get_latest_analysis(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Get the latest analysis configuration for a file."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id, time_column, x_column, y_column, z_column, t_column
                FROM analysis
                WHERE file_id = ?
                ORDER BY analysis_time DESC
                LIMIT 1
                """,
                (file_id,)
            )
            result = cursor.fetchone()
            
            if result is None:
                return None
            
            return {
                'analysis_id': result[0],
                'mapping': {
                    'time': result[1],
                    'x': result[2],
                    'y': result[3],
                    'z': result[4],
                    't': result[5]
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving latest analysis: {str(e)}")
            raise
    
    def get_analysis_results(self, analysis_id: int) -> Dict[str, Dict[str, Any]]:
        """Get all results for an analysis."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT axis, has_seasonal, amplitude, is_consistent,
                       summer_avg, winter_avg, is_progressive, slope,
                       r_squared, seasonal_strength, movement_type
                FROM results
                WHERE analysis_id = ?
                """,
                (analysis_id,)
            )
            results = cursor.fetchall()
            
            analysis_results = {}
            for row in results:
                analysis_results[row[0]] = {
                    'has_seasonal': bool(row[1]),
                    'amplitude': float(row[2]) if row[2] is not None else None,
                    'is_consistent': bool(row[3]),
                    'summer_avg': float(row[4]) if row[4] is not None else None,
                    'winter_avg': float(row[5]) if row[5] is not None else None,
                    'is_progressive': bool(row[6]),
                    'slope': float(row[7]) if row[7] is not None else None,
                    'r_squared': float(row[8]) if row[8] is not None else None,
                    'seasonal_strength': float(row[9]) if row[9] is not None else None,
                    'movement_type': row[10]
                }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error retrieving analysis results: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("Database connection closed")

# Create a global instance
db = Database() 