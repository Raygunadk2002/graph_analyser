import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import HTTPException
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.file_path: Optional[Path] = None
        self.mapping: Optional[Dict[str, str]] = None
        self.processed_data: Optional[Dict[str, Any]] = None
        
    def load_file(self, file_path: Path) -> None:
        """Load and preprocess the data file."""
        try:
            logger.debug(f"Loading file: {file_path}")
            
            # Read the file based on its extension
            if file_path.suffix.lower() == '.csv':
                self.df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                try:
                    # First try with openpyxl
                    self.df = pd.read_excel(file_path, engine='openpyxl')
                except Exception as openpyxl_error:
                    logger.warning(f"openpyxl error: {str(openpyxl_error)}, trying xlrd")
                    try:
                        self.df = pd.read_excel(file_path, engine='xlrd')
                    except Exception as xlrd_error:
                        logger.error(f"Both openpyxl and xlrd failed to read the file. Openpyxl error: {str(openpyxl_error)}, xlrd error: {str(xlrd_error)}")
                        raise HTTPException(
                            status_code=400,
                            detail="The Excel file appears to be corrupted or not in a valid format. Please ensure it's a valid Excel file and try again."
                        )
            
            # Validate that we got a DataFrame with data
            if self.df is None or self.df.empty:
                raise HTTPException(
                    status_code=400,
                    detail="The file appears to be empty or could not be read properly."
                )
            
            self.file_path = file_path
            logger.debug(f"File loaded successfully: {self.df.shape} rows, {self.df.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error loading file: {str(e)}")
    
    def set_mapping(self, mapping: Dict[str, str]) -> None:
        """Set the column mapping and preprocess the data."""
        try:
            logger.debug(f"Setting mapping: {mapping}")
            self.mapping = mapping
            
            if self.df is None:
                raise HTTPException(status_code=400, detail="No file data available. Please upload a file first.")
            
            # Get the mapped columns
            time_col = mapping.get('time')
            if not time_col:
                raise HTTPException(status_code=400, detail="No time column specified in mapping")
            
            # Convert time column to datetime and sort
            try:
                logger.debug(f"Converting time column '{time_col}' to datetime")
                self.df['__time__'] = pd.to_datetime(self.df[time_col])
                logger.debug(f"Sorting by time column")
                self.df.sort_values('__time__', inplace=True)
                logger.debug(f"Successfully converted and sorted time column. First few dates: {self.df['__time__'].head().tolist()}")
            except Exception as e:
                logger.error(f"Error converting time column: {str(e)}\n{traceback.format_exc()}")
                raise HTTPException(status_code=400, detail=f"Error converting time column: {str(e)}")
            
            # Prepare processed data
            logger.debug("Preparing processed data")
            self.processed_data = {
                'columns': self.df.columns.tolist(),
                'numeric_columns': self.df.select_dtypes(include=[np.number]).columns.tolist(),
                'datetime_column': time_col,
                'mapped_columns': {
                    'time': time_col,
                    'x': mapping.get('x'),
                    'y': mapping.get('y'),
                    'z': mapping.get('z'),
                    't': mapping.get('t')
                }
            }
            
            # Prepare plot data
            logger.debug("Preparing plot data")
            dates_str = self.df['__time__'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
            plot_data = {}
            
            for axis, col in self.processed_data['mapped_columns'].items():
                if col and col in self.df.columns:
                    try:
                        logger.debug(f"Processing plot data for axis {axis} ({col})")
                        plot_data[axis.upper()] = {
                            'time': dates_str,
                            'values': self.df[col].astype(float).tolist()
                        }
                    except Exception as e:
                        logger.error(f"Error preparing plot data for {axis}: {str(e)}\n{traceback.format_exc()}")
                        continue
            
            self.processed_data['plot_data'] = plot_data
            logger.debug("Data preprocessing completed successfully")
            
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error in data preprocessing: {str(e)}")
    
    def get_processed_data(self) -> Dict[str, Any]:
        """Get the processed data."""
        if self.processed_data is None:
            raise HTTPException(status_code=400, detail="No processed data available. Please set column mapping first.")
        return self.processed_data
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get the processed DataFrame."""
        if self.df is None:
            raise HTTPException(status_code=400, detail="No data available. Please upload a file first.")
        return self.df
    
    def get_mapping(self) -> Dict[str, str]:
        """Get the current column mapping."""
        if self.mapping is None:
            raise HTTPException(status_code=400, detail="No mapping available. Please set column mapping first.")
        return self.mapping

# Create a global instance
data_loader = DataLoader() 