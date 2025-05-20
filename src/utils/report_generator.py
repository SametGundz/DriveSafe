#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scientific Report Generator

This module provides comprehensive functionality for generating publication-quality,
scientific reports for driver drowsiness detection analysis. It includes:

- PDF report generation with sections, figures, and tables
- Statistical summary generation with relevant metrics
- Data visualization through high-quality matplotlib figures
- Session-based organization of all analysis products
- Export functionality to multiple formats (PDF, XLSX, CSV, JSON)

The module implements best practices from scientific publications to ensure
reports are suitable for academic or industrial presentations.
"""

# Set matplotlib backend to Agg (non-interactive) before importing matplotlib
import matplotlib
matplotlib.use('Agg')

import os
import csv
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import time
import tempfile

# Optional PDF report generation with ReportLab if available
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)

# Singleton pattern için bir global oturum dizini
_current_session_dir = None
_current_session_date_str = None
_current_session_paths = None

def get_or_create_current_session():
    """
    Mevcut oturum klasörünü döndür veya yeni bir tane oluştur.
    
    Returns:
        Tuple[str, str, Dict[str, str]]: (session_dir, date_str, paths)
    """
    global _current_session_dir, _current_session_date_str, _current_session_paths
    
    if _current_session_dir is None:
        # Yeni bir oturum oluştur
        report_generator = ScientificReportGenerator()
        
        # Timestamp-based folder oluştur (analysis_session prefix olmadan)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Base output directory
        base_output_dir = os.path.join(os.getcwd(), 'reports')
        
        # Tarih klasörü oluştur
        date_folder = now.strftime("%Y-%m-%d")  # Format: 2025-02-02
        date_dir = os.path.join(base_output_dir, date_folder)
        
        # Oturum dizini (tarih_saat)
        session_dir = os.path.join(date_dir, date_str)
        
        # Ana dizinleri oluştur
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(date_dir, exist_ok=True)
        os.makedirs(session_dir, exist_ok=True)
        
        # Oturum değişkenlerini ayarla
        _current_session_dir = session_dir
        _current_session_date_str = date_str
        _current_session_paths = {"root": session_dir}
        
        logger.info(f"Created new session folder: {_current_session_dir}")
    
    return _current_session_dir, _current_session_date_str, _current_session_paths

def reset_current_session():
    """
    Mevcut oturumu sıfırla, böylece yeni bir tane oluşturulabilir.
    """
    global _current_session_dir, _current_session_date_str, _current_session_paths
    _current_session_dir = None
    _current_session_date_str = None
    _current_session_paths = None
    logger.info("Reset current session")

class ScientificReportGenerator:
    """
    A comprehensive scientific report generator for drowsiness detection analysis.
    
    This class centralizes report generation functions and ensures consistent
    academic-quality outputs for all analysis products.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the report generator with configuration.
        
        Args:
            config: Optional configuration dictionary with report settings
        """
        self.config = config or {}
        
        # Default configuration values
        self.default_config = {
            'report_style': 'scientific',  # scientific, clinical, technical
            'page_size': 'A4',             # A4, letter
            'font_family': 'Helvetica',    # Helvetica, Times
            'title_font_size': 16,
            'section_font_size': 12,
            'body_font_size': 10,
            'include_timestamp': True,
            'dpi': 300,                    # Figure resolution
            'color_palette': 'viridis'     # seaborn/matplotlib palette
        }
        
        # Apply defaults for missing values
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Set up the visual style for plots
        self._setup_plot_style()
        
        logger.info("Scientific Report Generator initialized")
    
    def _setup_plot_style(self):
        """Configure the matplotlib/seaborn plotting style for consistency."""
        # Set the aesthetic style of the plots
        sns.set_style('whitegrid')
        
        # Use the specified color palette
        sns.set_palette(self.config['color_palette'])
        
        # Set up figure aesthetics for scientific publication
        plt.rcParams.update({
            'figure.figsize': (8, 6),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': self.config['dpi'],
            'savefig.dpi': self.config['dpi'],
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def create_session_folder(self, prefix: str = "session") -> Tuple[str, str, Dict[str, str]]:
        """
        Create a structured session folder for organizing reports
        with a prefix and timestamp.
        
        Args:
            prefix: Folder name prefix (default: "session")
            
        Returns:
            Tuple containing:
                - session_dir: Path to the created session directory
                - date_str: Formatted date string
                - paths: Dictionary of subdirectory paths (raw_data, figures, tables)
        """
        try:
            # Create a timestamp-based folder name
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d_%H-%M-%S")
            timestamp = now.timestamp()
            
            # Base output directory from configuration
            base_output_dir = self.config.get('output_dir', os.path.join(os.getcwd(), 'reports'))
            
            # Create a directory with year/month/day structure
            date_folder = now.strftime("%Y-%m-%d")  # Format: 2025-02-02
            date_dir = os.path.join(base_output_dir, date_folder)
            
            # Create the session directory within the date folder
            session_name = f"{date_str}"  # Just use the timestamp for the session folder
            timestamp_dir = os.path.join(date_dir, session_name)
            
            # Create analysis type directory (distraction_analysis or gaze_analysis)
            analysis_dir = os.path.join(timestamp_dir, prefix)
            
            # Ensure all parent directories exist
            os.makedirs(base_output_dir, exist_ok=True)  # Base reports dir
            os.makedirs(date_dir, exist_ok=True)         # Date dir
            os.makedirs(timestamp_dir, exist_ok=True)    # Session timestamp dir
            os.makedirs(analysis_dir, exist_ok=True)     # Analysis type dir
            
            # Create subdirectories for this analysis
            raw_data_dir = os.path.join(analysis_dir, "raw_data")
            os.makedirs(raw_data_dir, exist_ok=True)
            logger.info(f"Created raw_data folder at: {raw_data_dir}")
            
            paths = {
                "raw_data": raw_data_dir,
                "root": analysis_dir
            }
            
            # Distraction_analysis değilse figures ve tables klasörlerini oluştur
            if prefix != "distraction_analysis":
                figures_dir = os.path.join(analysis_dir, "figures")
                tables_dir = os.path.join(analysis_dir, "tables")
                
                os.makedirs(figures_dir, exist_ok=True)
                os.makedirs(tables_dir, exist_ok=True)
                
                logger.info(f"Created figures folder at: {figures_dir}")
                logger.info(f"Created tables folder at: {tables_dir}")
                
                paths["figures"] = figures_dir
                paths["tables"] = tables_dir
            
            logger.info(f"Created analysis folder at: {analysis_dir}")
            
            return analysis_dir, date_str, paths
            
        except Exception as e:
            logger.error(f"Error creating session folder: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # En azından bir basic dir oluşturmaya çalış
            fallback_dir = os.path.join(os.getcwd(), f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(fallback_dir, exist_ok=True)
            return fallback_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), {"root": fallback_dir}
    
    def save_json_report(self, data: Dict[Any, Any], file_path: str, pretty_print: bool = True) -> str:
        """
        Save data as JSON report with formatting options.
        
        Args:
            data: Dictionary to save as JSON
            file_path: Path to save the JSON file
            pretty_print: Whether to format the JSON with indentation
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Alt klasör tipini belirle
            if os.path.basename(os.path.dirname(file_path)) != "raw_data":
                # Eğer raw_data subdirectory içinde değilse kontrol et ve gerekirse düzelt
                session_dir = os.path.dirname(file_path)
                if os.path.exists(os.path.join(session_dir, "raw_data")):
                    # raw_data alt klasörü varsa, dosyayı oraya kaydet
                    new_path = os.path.join(session_dir, "raw_data", os.path.basename(file_path))
                    file_path = new_path
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    logger.info(f"JSON path updated to use raw_data subdirectory: {file_path}")
            
            # Convert numpy arrays/dataframes to native Python types
            class NumpyJSONEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict('records')
                    if isinstance(obj, pd.Series):
                        return obj.to_dict()
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    return super().default(obj)
            
            # Write to JSON
            with open(file_path, 'w') as f:
                if pretty_print:
                    json.dump(data, f, indent=4, cls=NumpyJSONEncoder)
                else:
                    json.dump(data, f, ensure_ascii=False)
                
            logger.info(f"JSON report saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving JSON report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def save_csv_report(self, data: Union[List[List[Any]], pd.DataFrame], 
                       headers: Optional[List[str]], file_path: str) -> str:
        """
        Save data as CSV report with support for both list data and pandas DataFrames.
        
        Args:
            data: List of rows or pandas DataFrame
            headers: Column headers (only used if data is a list)
            file_path: Path to save the CSV file
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Alt klasör tipini belirle
            if os.path.basename(os.path.dirname(file_path)) != "raw_data":
                # Eğer raw_data subdirectory içinde değilse kontrol et ve gerekirse düzelt
                session_dir = os.path.dirname(file_path)
                if os.path.exists(os.path.join(session_dir, "raw_data")):
                    # raw_data alt klasörü varsa, dosyayı oraya kaydet
                    new_path = os.path.join(session_dir, "raw_data", os.path.basename(file_path))
                    file_path = new_path
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    logger.info(f"CSV path updated to use raw_data subdirectory: {file_path}")
            
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if headers:
                        writer.writerow(headers)
                    writer.writerows(data)
            
            logger.info(f"CSV report saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving CSV report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def save_excel_report(self, data_dict: Dict[str, Union[List[List[Any]], pd.DataFrame]], 
                         file_path: str) -> str:
        """
        Save multiple datasets to different sheets of an Excel file.
        
        Args:
            data_dict: Dictionary mapping sheet names to data (lists or DataFrames)
            file_path: Path to save the Excel file
            
        Returns:
            str: Path to the saved file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Alt klasör tipini belirle
            if os.path.basename(os.path.dirname(file_path)) != "tables":
                # Eğer tables subdirectory içinde değilse kontrol et ve gerekirse düzelt
                session_dir = os.path.dirname(file_path)
                if os.path.exists(os.path.join(session_dir, "tables")):
                    # tables alt klasörü varsa, dosyayı oraya kaydet
                    new_path = os.path.join(session_dir, "tables", os.path.basename(file_path))
                    file_path = new_path
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    logger.info(f"Excel path updated to use tables subdirectory: {file_path}")
            
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                for sheet_name, data in data_dict.items():
                    if isinstance(data, pd.DataFrame):
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        # Convert list data to DataFrame
                        if isinstance(data, list) and len(data) > 0:
                            if isinstance(data[0], list):
                                # Check if first item is a list of column names
                                if all(isinstance(x, str) for x in data[0]):
                                    df = pd.DataFrame(data[1:], columns=data[0])
                                else:
                                    df = pd.DataFrame(data)
                            else:
                                df = pd.DataFrame(data)
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                # Access the XlsxWriter workbook and worksheet objects
                workbook = writer.book
                
                # Add a format for the header cells
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'border': 1,
                    'bg_color': '#D9E1F2',  # Light blue background
                    'font_name': 'Arial',
                    'font_size': 10
                })
                
                # Format all worksheets
                for sheet_name in data_dict.keys():
                    try:
                        worksheet = writer.sheets[sheet_name]
                        # Set column widths
                        worksheet.set_column(0, 0, 15)  # First column
                        worksheet.set_column(1, 10, 12)  # Rest of columns
                        
                        # Daha güvenli başlık formatlaması - doğrudan satır 0'daki hücreleri formatlıyoruz
                        if hasattr(worksheet, 'table') and hasattr(worksheet.table, 'columns'):
                            # Eski yöntem - worksheet.table.columns varsa kullan
                            for col_num, value in enumerate(worksheet.table.columns):
                                worksheet.write(0, col_num, value, header_format)
                        else:
                            # Data frame'in sütun sayısını tespit et ve ona göre formatla
                            df = data_dict[sheet_name]
                            if isinstance(df, pd.DataFrame):
                                for col_num, value in enumerate(df.columns):
                                    worksheet.write(0, col_num, value, header_format)
                    except Exception as e:
                        logger.warning(f"Could not format headers for sheet {sheet_name}: {str(e)}")
                        continue
            
            logger.info(f"Excel report saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving Excel report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def create_heatmap_figure(self, 
                            data_matrix: np.ndarray, 
                            row_labels: List[str], 
                            col_labels: List[str],
                            title: str, 
                            xlabel: str = "",
                            ylabel: str = "",
                            cmap: str = "viridis",
                            figsize: Tuple[int, int] = (10, 8),
                            annotate: bool = True) -> Figure:
        """
        Create a publication-quality heatmap.
        
        Args:
            data_matrix: 2D numpy array of data values
            row_labels: Labels for rows
            col_labels: Labels for columns
            title: Figure title
            xlabel: X-axis label
            ylabel: Y-axis label
            cmap: Colormap to use
            figsize: Figure size (width, height) in inches
            annotate: Whether to annotate cells with values
            
        Returns:
            Figure: Matplotlib figure object
        """
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        heatmap = sns.heatmap(
            data_matrix, 
            annot=annotate, 
            fmt=".2f" if np.any(np.mod(data_matrix, 1)) else ".0f", 
            linewidths=0.5, 
            ax=ax,
            cmap=cmap,
            cbar_kws={"shrink": 0.8}
        )
        
        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        
        # Set tick labels
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
        ax.set_yticklabels(row_labels, rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_pie_chart(self, 
                       labels: List[str], 
                       sizes: List[float], 
                       title: str, 
                       figsize: Tuple[int, int] = (8, 8),
                       colors: Optional[List[str]] = None,
                       explode: Optional[List[float]] = None,
                       text_kwargs: Optional[Dict[str, Any]] = None) -> Figure:
        """
        Create a publication-quality pie chart.
        
        Args:
            labels: Labels for pie chart segments
            sizes: Sizes for pie chart segments
            title: Chart title
            figsize: Figure size (width, height) in inches
            colors: Optional list of colors for segments
            explode: Optional list of explosion values for segments
            text_kwargs: Optional kwargs for text formatting
            
        Returns:
            Figure: Matplotlib figure object
        """
        try:
            # Check if data exists
            if not sizes:
                logger.warning("No data available for creating pie chart")
                return None
            
            # Default text options with scientific styling
            if text_kwargs is None:
                text_kwargs = {
                    'fontsize': 11,
                    'fontweight': 'bold',
                    'color': 'black'
                }
                
            # Create figure and axes
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create pie chart with scientific styling
            wedges, texts, autotexts = ax.pie(
                sizes, 
                labels=None,  # We'll add labels in the legend
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                colors=colors,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                textprops=text_kwargs
            )
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Add a legend with the labels
            ax.legend(
                wedges, 
                labels,
                title="Categories",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            # Add title with scientific styling
            ax.set_title(title, fontsize=14, pad=20)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            # Print stack trace for debugging
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def create_time_series_figure(self,
                                time_data: List[Union[float, datetime]],
                                series_data: Dict[str, List[float]],
                                title: str,
                                ylabel: str,
                                xlabel: str = "Time",
                                figsize: Tuple[int, int] = (12, 6),
                                markers: Optional[Dict[str, str]] = None,
                                threshold_lines: Optional[Dict[str, Tuple[float, str, str]]] = None) -> Figure:
        """
        Create a publication-quality time series plot with multiple series.
        
        Args:
            time_data: List of timestamps (either numeric or datetime objects)
            series_data: Dictionary mapping series names to data values
            title: Plot title
            ylabel: Y-axis label
            xlabel: X-axis label
            figsize: Figure size (width, height) in inches
            markers: Optional dictionary mapping series names to marker styles
            threshold_lines: Optional dictionary mapping labels to (value, color, style) tuples
            
        Returns:
            Figure: Matplotlib figure object
        """
        # Convert datetime objects if needed
        x_data = time_data
        is_datetime = False
        if time_data and isinstance(time_data[0], datetime):
            is_datetime = True
        
        # Create figure and axes with scientific styling
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each series
        for series_name, y_values in series_data.items():
            marker = markers.get(series_name, None) if markers else None
            ax.plot(x_data, y_values, label=series_name, marker=marker, linewidth=2)
        
        # Add threshold lines if specified
        if threshold_lines:
            for label, (value, color, style) in threshold_lines.items():
                ax.axhline(y=value, color=color, linestyle=style, linewidth=1.5, alpha=0.7)
                # Add text label for the threshold line
                ax.text(
                    0.02, 
                    value + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                    label, 
                    transform=ax.get_yaxis_transform(),
                    color=color, 
                    fontsize=9
                )
        
        # Configure axes
        if is_datetime:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()  # Rotate date labels
        
        # Add labels and title with scientific styling
        ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend with scientific styling
        ax.legend(
            loc='best', 
            frameon=True, 
            fancybox=True, 
            framealpha=0.8,
            edgecolor='gray'
        )
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def save_figure(self, figure: Figure, file_path: str, dpi: int = 300) -> str:
        """
        Save a matplotlib figure with publication-quality settings
        
        Args:
            figure: matplotlib figure to save
            file_path: Path to save the figure, will be created if necessary
            dpi: Resolution in dots per inch (default 300)
            
        Returns:
            str: Path to the saved figure
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Alt klasör tipini belirle
            if os.path.basename(os.path.dirname(file_path)) != "figures":
                # Eğer figures subdirectory içinde değilse kontrol et ve gerekirse düzelt
                session_dir = os.path.dirname(file_path)
                if os.path.exists(os.path.join(session_dir, "figures")):
                    # figures alt klasörü varsa, dosyayı oraya kaydet
                    new_path = os.path.join(session_dir, "figures", os.path.basename(file_path))
                    file_path = new_path
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    logger.info(f"Figure path updated to use figures subdirectory: {file_path}")
            
            # Set figure style for publication quality
            figure.set_tight_layout(True)
            figure.savefig(file_path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)  # Close the figure to free up memory
            logger.info(f"Figure saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving figure: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def generate_gaze_statistics_report(self, 
                                      stats: Dict[int, Dict[str, Any]], 
                                      session_name: Optional[str] = None,
                                      include_pdf: bool = True,
                                      include_raw_data: bool = True) -> Dict[str, str]:
        """
        Generate comprehensive gaze statistics report with visualizations.
        
        Args:
            stats: Dictionary of gaze statistics by zone ID
            session_name: Optional name for the session
            include_pdf: Whether to include PDF report
            include_raw_data: Whether to include raw data exports
            
        Returns:
            Dict[str, str]: Paths to generated report files
        """
        try:
            # Create session folder
            analysis_dir, date_str, paths = self.create_session_folder(prefix="gaze_analysis")
            
            # File paths
            report_paths = {}
            csv_file = os.path.join(paths['tables'], f"gaze_stats_{date_str}.csv")
            excel_file = os.path.join(paths['tables'], f"gaze_stats_{date_str}.xlsx")
            json_file = os.path.join(paths['raw_data'], f"gaze_stats_{date_str}.json")
            pie_file = os.path.join(paths['figures'], f"gaze_pie_{date_str}.png")
            # Ekstra olarak visit count için second pie chart
            visits_pie_file = os.path.join(paths['figures'], f"gaze_visits_pie_{date_str}.png")
            pdf_file = os.path.join(analysis_dir, f"gaze_analysis_report_{date_str}.pdf")
            
            # Prepare data for CSV and Excel
            headers = ["Zone ID", "Zone Name", "Duration (s)", "Visits", "Percentage (%)"]
            rows = []
            
            # Prepare data for raw export and visualization
            labels = []
            durations = []
            percentages = []
            zone_names = []
            visit_counts = []
            
            # Process stats data
            total_duration = 0.0
            for zone_id, zone_data in stats.items():
                # Calculate total duration
                zone_duration = zone_data.get('total_duration', 0.0)
                total_duration += zone_duration
            
            # Create rows with percentage calculation
            for zone_id, zone_data in sorted(stats.items()):
                zone_name = zone_data.get('name', f"Zone {zone_id}")
                zone_duration = zone_data.get('total_duration', 0.0)
                visit_count = zone_data.get('visit_count', 0)
                
                # Calculate percentage based on total duration
                percentage = 0.0
                if total_duration > 0:
                    percentage = (zone_duration / total_duration) * 100.0
                
                # Add to rows for tabular data
                rows.append([
                    str(zone_id),
                    zone_name,
                    f"{zone_duration:.2f}",
                    str(visit_count),
                    f"{percentage:.2f}"
                ])
                
                # Add to lists for visualization
                labels.append(zone_name)
                durations.append(zone_duration)
                percentages.append(percentage)
                zone_names.append(zone_name)
                visit_counts.append(visit_count)
            
            # Create CSV export if requested
            if include_raw_data:
                self.save_csv_report(rows, headers, csv_file)
                report_paths['csv'] = csv_file
                
                # Create Excel export
                excel_data = {"Gaze Statistics": rows}
                self.save_excel_report(excel_data, excel_file)
                report_paths['excel'] = excel_file
                
                # Create JSON export (more detailed)
                self.save_json_report(stats, json_file)
                report_paths['json'] = json_file
            
            # Create pie chart visualization - sadece pie chart yapalım, heatmap kaldıralım
            if len(durations) > 0:
                try:
                    # Bölge renklerini tanımlayalım
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    
                    # Create and save the pie chart for durations
                    fig = self.create_pie_chart(
                        labels=labels,
                        sizes=durations,
                        title="Gaze Duration by Zone",
                        figsize=(9, 6),
                        colors=colors
                    )
                    self.save_figure(fig, pie_file)
                    report_paths['pie_chart'] = pie_file
                    
                    # Create separate pie chart for visit counts
                    visit_fig = self.create_pie_chart(
                        labels=labels,
                        sizes=visit_counts,
                        title="Gaze Zone Visits",
                        figsize=(9, 6),
                        colors=colors
                    )
                    self.save_figure(visit_fig, visits_pie_file)
                    report_paths['visits_pie_chart'] = visits_pie_file
                    
                except Exception as e:
                    logger.error(f"Error creating data visualizations: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Generate PDF report if requested
            if include_pdf:
                try:
                    # Create PDF document
                    from reportlab.lib.pagesizes import A4
                    
                    doc = SimpleDocTemplate(
                        pdf_file,
                        pagesize=A4,
                        rightMargin=72,  # Sabit marj değeri kullan
                        leftMargin=72,   # Sabit marj değeri kullan 
                        topMargin=72,    # Sabit marj değeri kullan
                        bottomMargin=72  # Sabit marj değeri kullan
                    )
                    styles = getSampleStyleSheet()
                    
                    # Add content
                    content = []
                    
                    # Title
                    title = f"Gaze Analysis Report"
                    if session_name:
                        title += f" - {session_name}"
                    content.append(Paragraph(title, styles['Title']))
                    content.append(Spacer(1, 12))
                    
                    # Date
                    content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                           styles['Normal']))
                    content.append(Spacer(1, 12))
                    
                    # Summary statistics
                    content.append(Paragraph("Gaze Zone Statistics", styles['Heading2']))
                    content.append(Spacer(1, 6))
                    
                    # Add the CSV data as a table
                    table_data = [headers] + rows
                    table = Table(table_data, colWidths=[40, 120, 80, 60, 80])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    content.append(table)
                    
                    # Add spacer
                    content.append(Spacer(1, 12))
                    
                    # Add pie chart
                    if os.path.exists(pie_file):
                        content.append(Paragraph("Gaze Duration Distribution", styles['Heading2']))
                        content.append(Spacer(1, 6))
                        img = Image(pie_file, width=400, height=300)
                        content.append(img)
                        content.append(Spacer(1, 12))
                    
                    # Add visit counts pie chart
                    if os.path.exists(visits_pie_file):
                        content.append(Paragraph("Gaze Zone Visits Distribution", styles['Heading2']))
                        content.append(Spacer(1, 6))
                        img = Image(visits_pie_file, width=400, height=300)
                        content.append(img)
                        content.append(Spacer(1, 12))
                    
                    # Add footer
                    content.append(Paragraph("This report was automatically generated by the Driver Drowsiness Detection System.", 
                                           styles['Normal']))
                    
                    # Build PDF
                    doc.build(content)
                    report_paths['pdf'] = pdf_file
                    logger.info(f"PDF report generated: {pdf_file}")
                    
                except Exception as e:
                    logger.warning("PDF report generation failed - ReportLab not available")
                except ImportError:
                    logger.warning("PDF report generation failed - ReportLab not available")
                except Exception as e:
                    logger.error(f"Error generating PDF report: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            return report_paths
            
        except Exception as e:
            logger.error(f"Error generating gaze statistics report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def generate_distraction_report(self, 
                                 distraction_data: Dict[str, Any], 
                                 file_path: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Generate a distraction report from the provided data.
        
        Args:
            distraction_data: Distraction report data
            file_path: Optional specific path to save the report
                      (if None, a timestamped session folder is created)
                      
        Returns:
            Dict[str, str]: Paths to the generated report files, or None if error
        """
        try:
            report_paths = {}
            
            # Create a session folder if file_path not provided
            if file_path is None:
                # Specifically use "distraction_analysis" as the prefix for this type of analysis
                analysis_dir, date_str, paths = self.create_session_folder(prefix="distraction_analysis")
                
                # Sadece raw_data alt klasörü ve analiz klasörünün kendisini kullan
                json_path = os.path.join(paths['raw_data'], f"distraction_report_{date_str}.json")
                # PDF doğrudan distraction_analysis klasörüne kaydedilecek
                pdf_path = os.path.join(paths['root'], f"distraction_analysis_report_{date_str}.pdf")
            else:
                json_path = file_path
                pdf_path = file_path.replace('.json', '.pdf')
                # Ensure directory exists
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            # Save the distraction data as JSON
            self.save_json_report(distraction_data, json_path)
            report_paths['json'] = json_path
            logger.info(f"JSON distraction data saved to: {json_path}")
            
            # Create PDF report if reportlab is available
            try:
                # Create PDF document
                from reportlab.lib.pagesizes import A4
                
                doc = SimpleDocTemplate(
                    pdf_path,
                    pagesize=A4,
                    rightMargin=72,  # Sabit marj değeri kullan
                    leftMargin=72,   # Sabit marj değeri kullan 
                    topMargin=72,    # Sabit marj değeri kullan
                    bottomMargin=72  # Sabit marj değeri kullan
                )
                styles = getSampleStyleSheet()
                
                # Define custom styles - check if they already exist first
                custom_styles = {
                    'Title': {
                        'parent': styles['Heading1'],
                        'fontSize': 16,
                        'spaceAfter': 12
                    },
                    'Section': {
                        'parent': styles['Heading2'],
                        'fontSize': 14,
                        'spaceAfter': 10
                    },
                    'Normal': {
                        'parent': styles['Normal'],
                        'fontSize': 10,
                        'spaceAfter': 6
                    }
                }
                
                # Güvenli bir şekilde stilleri ekle
                for style_name, style_props in custom_styles.items():
                    if style_name not in styles:
                        styles.add(ParagraphStyle(
                            name=style_name,
                            parent=style_props['parent'],
                            fontSize=style_props['fontSize'],
                            spaceAfter=style_props['spaceAfter']
                        ))
                
                # Build the document content
                content = []
                
                # Title
                content.append(Paragraph("Driver Distraction Analysis Report", styles['Title']))
                content.append(Spacer(1, 12))
                
                # Generate date and time
                current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                content.append(Paragraph(f"Report Generated: {current_date}", styles['Normal']))
                content.append(Spacer(1, 12))
                
                # Summary section
                content.append(Paragraph("Summary", styles['Section']))
                
                # Extract summary data
                summary_data = []
                summary_data.append(["Metric", "Value"])
                
                # Tüm alanların kontrolü ve formatlanması
                metrics_to_include = [
                    ('total_driving_time', "Total Driving Time (s)"),
                    ('total_distraction_time', "Total Distraction Time (s)"),
                    ('area1_percentage', "Area 1 Percentage (%)"),
                    ('road_center_absence_time', "Road Center Absence Time (s)"),
                ]
                
                for key, label in metrics_to_include:
                    if key in distraction_data:
                        value = distraction_data[key]
                        if isinstance(value, (int, float)):
                            formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = str(value)
                        summary_data.append([label, formatted_value])
                
                # Alan1 yüzdesi özel hesaplama
                if 'total_driving_time' in distraction_data and 'total_distraction_time' in distraction_data:
                    if distraction_data['total_driving_time'] > 0:
                        distraction_pct = (distraction_data['total_distraction_time'] / distraction_data['total_driving_time']) * 100
                        summary_data.append(["Distraction Percentage (%)", f"{distraction_pct:.2f}"])
                
                # Toplam olay sayısı
                if 'distraction_events' in distraction_data:
                    summary_data.append(["Number of Distraction Events", len(distraction_data['distraction_events'])])
                
                # EU regulation uyumluluk bilgisi
                if 'eu_regulation' in distraction_data:
                    eu_reg = distraction_data['eu_regulation']
                    if 'overall_compliance' in eu_reg:
                        summary_data.append(["EU Regulation Compliance", eu_reg['overall_compliance']])
                
                # Create the summary table
                if len(summary_data) > 1:  # Only create table if we have data
                    table = Table(summary_data, colWidths=[250, 150])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    content.append(table)
                    content.append(Spacer(1, 12))
                
                # Zone Statistics section if available
                if 'zone_statistics' in distraction_data:
                    content.append(Paragraph("Zone Statistics", styles['Section']))
                    content.append(Spacer(1, 12))
                    
                    zone_data = [["Zone ID", "Zone Name", "Duration (s)", "Visits", "Is Area 1"]]
                    
                    for zone_id, zone_info in distraction_data['zone_statistics'].items():
                        zone_data.append([
                            zone_id,
                            zone_info.get('name', 'Unknown'),
                            f"{zone_info.get('total_duration', 0):.2f}",
                            zone_info.get('visit_count', 0),
                            "Yes" if zone_info.get('is_area1', False) else "No"
                        ])
                    
                    if len(zone_data) > 1:
                        zone_table = Table(zone_data, colWidths=[50, 120, 100, 70, 70])
                        zone_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)
                        ]))
                        content.append(zone_table)
                        content.append(Spacer(1, 12))
                
                # Events section
                if 'distraction_events' in distraction_data and distraction_data['distraction_events']:
                    content.append(Paragraph("Distraction Events", styles['Section']))
                    content.append(Spacer(1, 12))
                    
                    events_data = []
                    events_data.append(["#", "Start Time (s)", "Duration (s)", "Gaze Zone"])
                    
                    for idx, event in enumerate(distraction_data['distraction_events']):
                        # Daha güvenli bir şekilde değerleri kontrol et
                        start_time = event.get('start_time', 'N/A')
                        duration = event.get('duration', 'N/A')
                        gaze_zone = event.get('gaze_zone', 'N/A')
                        
                        # Sayısal değerleri formatla
                        if isinstance(start_time, (int, float)):
                            start_time = f"{start_time:.2f}"
                        if isinstance(duration, (int, float)):
                            duration = f"{duration:.2f}"
                            
                        events_data.append([idx + 1, start_time, duration, gaze_zone])
                    
                    # Create the events table
                    events_table = Table(events_data, colWidths=[40, 120, 120, 120])
                    events_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    content.append(events_table)
                    content.append(Spacer(1, 12))
                
                # Build document
                doc.build(content)
                report_paths['pdf'] = pdf_path
                logger.info(f"PDF distraction report saved to: {pdf_path}")
                
            except ImportError as e:
                logger.warning(f"Could not create PDF report - reportlab not available: {str(e)}")
            except Exception as e:
                logger.error(f"Error creating PDF report: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
            
            return report_paths
        
        except Exception as e:
            logger.error(f"Error generating distraction report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def gaze_statistics_to_df(self, gaze_stats: Dict[str, Any]) -> pd.DataFrame:
        """
        Bakış istatistiklerini pandas DataFrame'e dönüştür.
        
        Args:
            gaze_stats: Bakış istatistikleri içeren sözlük
            
        Returns:
            pd.DataFrame: İstatistikleri içeren DataFrame
        """
        try:
            import pandas as pd
            logger.info(f"Converting gaze statistics to DataFrame using pandas {pd.__version__}")
            
            # Giriş verilerini detaylı logla
            logger.info(f"gaze_stats keys: {list(gaze_stats.keys())}")
            
            # Gerekli alanları kontrol et
            required_fields = ["zone_durations", "zone_percentages", "zone_visit_counts"]
            for field in required_fields:
                if field not in gaze_stats:
                    logger.error(f"Missing field in gaze_stats: {field}")
                    return pd.DataFrame(columns=['Zone', 'Duration (s)', 'Percentage (%)', 'Visit Count', 'Avg Duration (s)'])

            # Veriler boş mu kontrol et ve loglama yap
            if not gaze_stats["zone_durations"]:
                logger.warning("Zone durations dictionary is empty")
            else:
                logger.info(f"Zone durations: {gaze_stats['zone_durations']}")
                
            if not gaze_stats["zone_percentages"]:
                logger.warning("Zone percentages dictionary is empty")
            else:
                logger.info(f"Zone percentages: {gaze_stats['zone_percentages']}")
                
            if not gaze_stats["zone_visit_counts"]:
                logger.warning("Zone visit counts dictionary is empty")
            else:
                logger.info(f"Zone visit counts: {gaze_stats['zone_visit_counts']}")
            
            # DataFrame oluştur
            data = []
            for zone_name in sorted(gaze_stats["zone_durations"].keys()):
                try:
                    duration = float(gaze_stats["zone_durations"].get(zone_name, 0.0))
                    percentage = float(gaze_stats["zone_percentages"].get(zone_name, 0.0))
                    visit_count = int(gaze_stats["zone_visit_counts"].get(zone_name, 0))
                    
                    # Division by zero kontrolü
                    if visit_count > 0:
                        avg_duration = duration / visit_count
                    else:
                        avg_duration = 0.0
                    
                    # Geçerli değerler kontrolü
                    if duration < 0:
                        logger.warning(f"Negative duration for zone {zone_name}: {duration}, setting to 0")
                        duration = 0.0
                    if percentage < 0:
                        logger.warning(f"Negative percentage for zone {zone_name}: {percentage}, setting to 0")
                        percentage = 0.0
                    
                    data.append({
                        'Zone': zone_name,
                        'Duration (s)': duration,
                        'Percentage (%)': percentage,
                        'Visit Count': visit_count,
                        'Avg Duration (s)': avg_duration
                    })
                    logger.debug(f"Added row for zone {zone_name}: Duration={duration:.2f}s, Percentage={percentage:.2f}%, Visits={visit_count}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing zone {zone_name}: {e}")
                
            # DataFrame oluştur
            if data:
                logger.info(f"Creating DataFrame with {len(data)} rows")
                df = pd.DataFrame(data)
                
                # Verileri sırala - Duration'a göre azalan sırada
                df = df.sort_values(by='Duration (s)', ascending=False)
                
                # Metadata ekle
                if 'total_duration' in gaze_stats:
                    df.attrs['total_duration'] = float(gaze_stats['total_duration'])
                else:
                    df.attrs['total_duration'] = float(sum(gaze_stats['zone_durations'].values()))
                    
                if 'data_points' in gaze_stats:
                    df.attrs['data_points'] = int(gaze_stats['data_points'])
                else:
                    df.attrs['data_points'] = 0
                    
                if 'session_name' in gaze_stats:
                    df.attrs['session_name'] = str(gaze_stats['session_name'])
                else:
                    df.attrs['session_name'] = "Unknown"
                
                logger.info(f"DataFrame created with columns: {df.columns.tolist()}")
                logger.info(f"DataFrame row count: {len(df)}")
                
                return df
            else:
                logger.warning("No data rows created for DataFrame")
                return pd.DataFrame(columns=['Zone', 'Duration (s)', 'Percentage (%)', 'Visit Count', 'Avg Duration (s)'])
                
        except Exception as e:
            logger.error(f"Error converting gaze stats to DataFrame: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Boş DataFrame döndür
            return pd.DataFrame(columns=['Zone', 'Duration (s)', 'Percentage (%)', 'Visit Count', 'Avg Duration (s)'])

    def gaze_statistics_report_to_pdf(self, gaze_stats: Dict[str, Any], output_path: str) -> Optional[str]:
        """
        Bakış istatistiklerini PDF raporuna dönüştür.
        
        Args:
            gaze_stats: Bakış istatistiklerini içeren sözlük
            output_path: Çıktı PDF dosyasının yolu
            
        Returns:
            str: Oluşturulan PDF dosyasının yolu, veya None (hata durumunda)
        """
        try:
            # İstatistikleri DataFrame'e dönüştür
            stats_df = self.gaze_statistics_to_df(gaze_stats)
            if stats_df.empty:
                logger.error("Cannot create report from empty statistics")
                return None
            
            # Figures ve tables klasör yollarını belirle
            report_dir = os.path.dirname(output_path)
            figures_dir = os.path.join(report_dir, "figures")
            tables_dir = os.path.join(report_dir, "tables")
            
            # Klasörlerin var olduğundan emin ol
            os.makedirs(figures_dir, exist_ok=True)
            os.makedirs(tables_dir, exist_ok=True)
            
            # Timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # PDF belgesi oluştur
            from reportlab.lib.pagesizes import A4
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,  # Sabit marj değeri kullan
                leftMargin=72,   # Sabit marj değeri kullan 
                topMargin=72,    # Sabit marj değeri kullan
                bottomMargin=72  # Sabit marj değeri kullan
            )
            
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='ReportTitle',  # 'Title' adı yerine 'ReportTitle' kullanıldı
                fontName=self.config.get('font_family', 'Helvetica-Bold'),
                fontSize=16,
                alignment=1,
                spaceAfter=12
            ))
            
            # Bilimsel açıklama stili ekle
            styles.add(ParagraphStyle(
                name='FigureDescription',
                fontName=self.config.get('font_family', 'Helvetica-Oblique'),
                fontSize=10,
                alignment=1,
                spaceAfter=12,
                textColor=colors.darkblue
            ))
            
            story = []
            
            # Rapor başlığı
            title = Paragraph("Gaze Statistics Report", styles['ReportTitle'])  # Stil adı burada da güncellendi
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Özet bilgiler
            session_name = stats_df.attrs.get('session_name', 'Unknown')
            total_duration = stats_df.attrs.get('total_duration', 0)
            data_points = stats_df.attrs.get('data_points', 0)
            
            summary_text = f"""
            <b>Session:</b> {session_name}<br/>
            <b>Total Duration:</b> {total_duration:.2f} seconds<br/>
            <b>Data Points:</b> {data_points}<br/>
            <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            summary = Paragraph(summary_text, styles['Normal'])
            story.append(summary)
            story.append(Spacer(1, 12))
            
            # Ana tablo
            main_table_data = [['Zone', 'Duration (s)', 'Percentage (%)', 'Visit Count', 'Avg Duration (s)']]
            # DataFrame'i tabloya dönüştür
            for _, row in stats_df.iterrows():
                main_table_data.append([
                    row['Zone'],
                    f"{row['Duration (s)']:.2f}",
                    f"{row['Percentage (%)']:.2f}%",
                    str(row['Visit Count']),
                    f"{row['Avg Duration (s)']:.2f}"
                ])
                
            main_table = Table(main_table_data)
            main_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(main_table)
            story.append(Spacer(1, 24))
            
            # Figürleri ara ve ekle
            figure_files = []
            pie_chart_file = None
            bar_chart_file = None
            
            # Figür klasöründeki dosyaları kontrol et
            if os.path.exists(figures_dir):
                for file in os.listdir(figures_dir):
                    file_path = os.path.join(figures_dir, file)
                    if file.endswith(('.png', '.jpg')):
                        figure_files.append(file_path)
                        if 'pie' in file:
                            pie_chart_file = file_path
                        elif 'bar' in file or 'durations' in file:
                            bar_chart_file = file_path
            
            # Eğer hiç figür oluşturulmamışsa, manuel olarak oluştur
            if not figure_files:
                try:
                    logger.info("No figures found. Creating figures manually...")
                    
                    # matplotlib'ı import et
                    import matplotlib
                    matplotlib.use('Agg')  # GUI olmayan ortamlar için
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    # Dizinin var olduğundan emin ol
                    os.makedirs(figures_dir, exist_ok=True)
                    
                    # zone_percentages'tan pasta grafiği oluştur
                    zone_names = list(gaze_stats["zone_percentages"].keys())
                    percentages = list(gaze_stats["zone_percentages"].values())
                    
                    # Sıfır olmayan değerleri filtrele (0.1'den büyük yüzdeler)
                    filtered_data = [(name, pct) for name, pct in zip(zone_names, percentages) if pct > 0.1]
                    if filtered_data:
                        filtered_zones, filtered_percentages = zip(*filtered_data)
                        
                        # Pasta grafik oluştur
                        plt.figure(figsize=(8, 6))
                        plt.pie(filtered_percentages, labels=filtered_zones, autopct='%1.1f%%', 
                                startangle=90, shadow=True)
                        plt.axis('equal')  # Daire şeklinde
                        plt.title('Gaze Distribution by Zone')
                        
                        # Kaydet
                        pie_file = os.path.join(figures_dir, f"gaze_pie_{date_str}.png")
                        plt.savefig(pie_file, dpi=300)
                        plt.close()
                        
                        figure_files.append(pie_file)
                        pie_chart_file = pie_file
                        logger.info(f"Created pie chart: {pie_file}")
                    
                    # Zone durations çubuk grafiği
                    durations = list(gaze_stats["zone_durations"].values())
                    
                    plt.figure(figsize=(10, 6))
                    plt.bar(zone_names, durations)
                    plt.title('Zone Durations')
                    plt.xlabel('Zone')
                    plt.ylabel('Duration (seconds)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    bar_file = os.path.join(figures_dir, f"zone_durations_bar_{date_str}.png")
                    plt.savefig(bar_file, dpi=300)
                    plt.close()
                    
                    figure_files.append(bar_file)
                    bar_chart_file = bar_file
                    logger.info(f"Created bar chart: {bar_file}")
                    
                except Exception as e:
                    logger.error(f"Error creating manual figures: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Figürler için bilimsel açıklamalar
            pie_chart_description = """
            <b>Figure 1: Gaze Distribution by Zone</b><br/>
            This pie chart illustrates the proportional distribution of the driver's gaze across different zones in the vehicle cockpit. 
            The visual representation quantifies attentional allocation patterns, highlighting dominant areas of visual focus.
            According to EU regulation C(2023)4523, excessive focus on non-driving areas (Area 1) may indicate distraction risk.
            """
            
            bar_chart_description = """
            <b>Figure 2: Zone Duration Analysis</b><br/>
            This bar chart presents the absolute duration (in seconds) the driver spent looking at each zone.
            Temporal analysis of gaze patterns provides insights into visual attention allocation strategies during driving.
            Extended durations in certain zones, particularly information displays (Infotainment), correlate with increased cognitive load
            and potential attentional disengagement from the primary driving task.
            """
            
            # Pasta grafiği ekle
            if pie_chart_file and os.path.exists(pie_chart_file):
                story.append(Paragraph("Gaze Distribution Analysis", styles['Heading2']))
                story.append(Spacer(1, 6))
                img = Image(pie_chart_file, width=400, height=300)
                story.append(img)
                story.append(Spacer(1, 6))
                story.append(Paragraph(pie_chart_description, styles['FigureDescription']))
                story.append(Spacer(1, 12))
            
            # Çubuk grafiği ekle
            if bar_chart_file and os.path.exists(bar_chart_file):
                story.append(Paragraph("Temporal Gaze Analysis", styles['Heading2']))
                story.append(Spacer(1, 6))
                img = Image(bar_chart_file, width=400, height=300)
                story.append(img)
                story.append(Spacer(1, 6))
                story.append(Paragraph(bar_chart_description, styles['FigureDescription']))
                story.append(Spacer(1, 12))
            
            # Tabloları tables klasörüne kaydet
            # 1. CSV tablosu
            csv_file_path = os.path.join(tables_dir, f"gaze_statistics_table_{timestamp}.csv")
            stats_df.to_csv(csv_file_path, index=False)
            logger.info(f"Gaze statistics CSV table saved to: {csv_file_path}")
            
            # 2. Excel tablosu
            try:
                excel_file_path = os.path.join(tables_dir, f"gaze_statistics_excel_{timestamp}.xlsx")
                
                # pandas'ı import et
                import pandas as pd
                
                # Excel Writer ile zengin formatlar ekle
                with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                    # Ana istatistikler sheet'i
                    stats_df.to_excel(writer, sheet_name='Zone Statistics', index=False)
                    
                    # Metadata sheet'i
                    metadata_df = pd.DataFrame({
                        'Field': ['Session Name', 'Total Duration (s)', 'Data Points', 'Report Date', 'Total Percentage'],
                        'Value': [
                            session_name,
                            f"{total_duration:.2f}",
                            str(data_points),
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            f"{sum(stats_df['Percentage (%)']):.2f}%"
                        ]
                    })
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    
                    # Workbook ve worksheet nesnelerine erişim
                    workbook = writer.book
                    
                    # Metadata sayfası
                    worksheet = writer.sheets['Metadata']
                    for idx, col in enumerate(['A', 'B']):
                        worksheet.column_dimensions[col].width = 20
                    
                    # Ana istatistikler sayfası
                    worksheet = writer.sheets['Zone Statistics']
                    for idx, col in enumerate(['A', 'B', 'C', 'D', 'E']):
                        worksheet.column_dimensions[col].width = 15
                
                logger.info(f"Gaze statistics Excel table saved to: {excel_file_path}")
            except Exception as e:
                logger.error(f"Error creating Excel table: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # PDF oluştur
            doc.build(story)
            
            logger.info(f"Gaze statistics PDF report created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating gaze statistics PDF report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def distraction_report_to_pdf(self, distraction_data: Dict[str, Any], output_path: str) -> Optional[str]:
        """
        Transform distraction data into a PDF report.
        
        Args:
            distraction_data: Dictionary containing distraction data
            output_path: Path to save the PDF
            
        Returns:
            str: Path to generated PDF, or None if error occurs
        """
        try:
            # Distraction data DataFrame
            df = self.distraction_data_to_df(distraction_data)
            if df.empty:
                logger.error("Cannot create report from empty data")
                return None
            
            # Create folders for figures if needed
            report_dir = os.path.dirname(output_path)
            figures_dir = os.path.join(report_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Get timestamp string
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Create PDF document
            from reportlab.lib.pagesizes import A4
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(
                name='ReportTitle',
                fontName=self.config.get('font_family', 'Helvetica-Bold'),
                fontSize=16,
                alignment=1,
                spaceAfter=12
            ))
            
            # Add description style
            styles.add(ParagraphStyle(
                name='FigureDescription',
                fontName=self.config.get('font_family', 'Helvetica-Oblique'),
                fontSize=10,
                alignment=1,
                spaceAfter=12,
                textColor=colors.darkblue
            ))
            
            content = []
            
            # Title
            title = Paragraph("Driver Distraction Analysis Report", styles['ReportTitle'])
            content.append(title)
            content.append(Spacer(1, 12))
            
            # Summary information
            summary_text = f"""
            <b>Session:</b> {df.attrs.get('session_name', 'Unknown')}<br/>
            <b>Total Duration:</b> {df.attrs.get('total_duration', 0):.2f} seconds<br/>
            <b>Distraction Events:</b> {df.attrs.get('event_count', 0)}<br/>
            <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            """
            summary = Paragraph(summary_text, styles['Normal'])
            content.append(summary)
            content.append(Spacer(1, 12))
            
            # Summary statistics
            if 'statistics' in distraction_data and distraction_data['statistics']:
                stats = distraction_data['statistics']
                
                stats_text = f"""
                <b>Average Distraction Duration:</b> {stats.get('avg_distraction_duration', 0):.2f} seconds<br/>
                <b>Max Distraction Duration:</b> {stats.get('max_distraction_duration', 0):.2f} seconds<br/>
                <b>Total Distraction Time:</b> {stats.get('total_distraction_time', 0):.2f} seconds<br/>
                <b>Distraction Percentage:</b> {stats.get('distraction_percentage', 0):.2f}%<br/>
                """
                stats_para = Paragraph(stats_text, styles['Normal'])
                content.append(stats_para)
                content.append(Spacer(1, 12))
            
            # Summary table
            if not df.empty:
                summary_table_data = [['Distraction Type', 'Count', 'Avg Duration (s)', 'Total Duration (s)', 'Percentage (%)']]
                
                # Group by distraction type and calculate statistics
                if 'events' in distraction_data and distraction_data['events']:
                    events = distraction_data['events']
                    
                    # Count distraction types
                    distraction_counts = {}
                    distraction_durations = {}
                    total_duration = 0
                    
                    for event in events:
                        event_type = event.get('type', 'Unknown')
                        duration = event.get('duration', 0)
                        
                        if event_type not in distraction_counts:
                            distraction_counts[event_type] = 0
                            distraction_durations[event_type] = 0
                        
                        distraction_counts[event_type] += 1
                        distraction_durations[event_type] += duration
                        total_duration += duration
                    
                    # Create table rows
                    for dist_type in distraction_counts:
                        count = distraction_counts[dist_type]
                        total_type_duration = distraction_durations[dist_type]
                        avg_duration = total_type_duration / count if count > 0 else 0
                        percentage = (total_type_duration / total_duration * 100) if total_duration > 0 else 0
                        
                        summary_table_data.append([
                            dist_type,
                            str(count),
                            f"{avg_duration:.2f}",
                            f"{total_type_duration:.2f}",
                            f"{percentage:.2f}%"
                        ])
                
                # Create table
                summary_table = Table(summary_table_data)
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                content.append(summary_table)
                content.append(Spacer(1, 24))
            
            # All distraction events table
            if 'events' in distraction_data and distraction_data['events']:
                events = distraction_data['events']
                
                content.append(Paragraph("Distraction Event Details", styles['Heading2']))
                content.append(Spacer(1, 6))
                
                events_table_data = [['Event ID', 'Type', 'Start Time', 'Duration (s)', 'Severity']]
                
                # Add each event to the table
                for i, event in enumerate(events):
                    event_id = event.get('id', i+1)
                    event_type = event.get('type', 'Unknown')
                    start_time = event.get('start_time', 0)
                    duration = event.get('duration', 0)
                    severity = event.get('severity', 'Low')
                    
                    # Format time as MM:SS
                    formatted_time = f"{int(start_time // 60):02d}:{int(start_time % 60):02d}"
                    
                    events_table_data.append([
                        str(event_id),
                        event_type,
                        formatted_time,
                        f"{duration:.2f}",
                        severity
                    ])
                
                # Create events table
                events_table = Table(events_table_data)
                events_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                content.append(events_table)
                content.append(Spacer(1, 12))
            
            # Distraction report başlık ve detaylar
            distraction_details = """
            <b>Understanding Distraction Events</b><br/>
            This report analyzes patterns of driver distraction during the driving session. Distraction events are 
            categorized based on duration, frequency, and severity levels. According to EU regulation C(2023)4523, 
            driver inattention exceeding 2 seconds significantly increases crash risk by up to 400%.
            <br/><br/>
            <b>Recommendations:</b><br/>
            - Monitor frequent distraction episodes and implement appropriate interventions<br/>
            - Evaluate whether in-vehicle technologies contribute to distraction events<br/>
            - Consider driver training on maintaining attention during complex driving situations<br/>
            """
            content.append(Paragraph(distraction_details, styles['Normal']))
            
            # Build the document
            doc.build(content)
            
            logger.info(f"Distraction report PDF generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating distraction report PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# Function to get a configured report generator
def get_report_generator(config: Optional[Dict[str, Any]] = None) -> ScientificReportGenerator:
    """
    Factory function to create and configure a ScientificReportGenerator instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ScientificReportGenerator: Configured report generator instance
    """
    return ScientificReportGenerator(config)

# Harici API fonksiyonları (backward compatibility için)
def generate_gaze_statistics_report(gaze_stats: Dict[str, Any], timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Bakış bölgesi istatistiklerinden rapor oluştur.
    
    Args:
        gaze_stats: Bakış istatistiklerini içeren sözlük.
        timestamp: İsteğe bağlı zaman damgası, belirtilmezse şu anki zaman kullanılır.
        
    Returns:
        Dict[str, Any]: Oluşturulan dosyaların yolları, veya None (hata durumunda).
    """
    try:
        # Matplotlib kurulumunu kontrol et
        try:
            import matplotlib
            matplotlib.use('Agg')  # GUI olmayan ortamlar için backend'i ayarla
            import matplotlib.pyplot as plt
            logger.info(f"Matplotlib version: {matplotlib.__version__}, backend: {matplotlib.get_backend()}")
        except ImportError as e:
            logger.error(f"Matplotlib not installed or error importing: {e}")
        except Exception as e:
            logger.error(f"Error configuring matplotlib: {e}")
            
        # Gaze istatistiklerini logla
        logger.info(f"Generating gaze statistics report with: {gaze_stats}")

        # İstatistik verilerini doğrula
        if not gaze_stats or not isinstance(gaze_stats, dict):
            logger.error("Invalid gaze statistics: empty or not a dictionary")
            return None

        # Gerekli alanların varlığını kontrol et
        required_fields = ["zone_durations", "zone_percentages", "zone_visit_counts"]
        for field in required_fields:
            if field not in gaze_stats:
                logger.error(f"Missing required field in gaze statistics: {field}")
                return None
                
        # İçerik kontrolü - her tablo için en az bir öğenin varlığını doğrula
        if not gaze_stats["zone_durations"]:
            logger.warning("Empty zone durations")
        if not gaze_stats["zone_percentages"]:
            logger.warning("Empty zone percentages")
        
        # Zone durationları kontrol et
        total_duration = 0
        if isinstance(gaze_stats["zone_durations"], dict):
            total_duration = sum(gaze_stats["zone_durations"].values())
            logger.info(f"Total duration from data: {total_duration:.2f} seconds")
            if total_duration < 0.001:
                logger.warning("Total duration is near zero - may cause issues with percentages")
        else:
            logger.warning(f"zone_durations has unexpected type: {type(gaze_stats['zone_durations'])}")
            
        # Yeni zaman damgası-bazlı klasör oluştur
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if timestamp is None else timestamp
        
        # Oturum klasörü oluştur
        session_dir, date_str, paths = get_or_create_current_session()
        
        # gaze_analysis için alt dizinler oluştur
        gaze_analysis_dir = os.path.join(session_dir, "gaze_analysis")
        os.makedirs(gaze_analysis_dir, exist_ok=True)
        
        gaze_raw_data_dir = os.path.join(gaze_analysis_dir, "raw_data")
        os.makedirs(gaze_raw_data_dir, exist_ok=True)
        
        gaze_figures_dir = os.path.join(gaze_analysis_dir, "figures")
        os.makedirs(gaze_figures_dir, exist_ok=True)
        
        gaze_tables_dir = os.path.join(gaze_analysis_dir, "tables")
        os.makedirs(gaze_tables_dir, exist_ok=True)
        
        # Dizinleri logla
        logger.info(f"Created directories for report:")
        logger.info(f"- Main directory: {gaze_analysis_dir}")
        logger.info(f"- Figures directory: {gaze_figures_dir}")
        logger.info(f"- Tables directory: {gaze_tables_dir}")
        logger.info(f"- Raw data directory: {gaze_raw_data_dir}")
        
        # Rapor dosya adları
        csv_file = os.path.join(gaze_raw_data_dir, f"gaze_statistics_{date_str}.csv")
        pdf_file = os.path.join(gaze_analysis_dir, f"gaze_statistics_report_{date_str}.pdf")
        
        # ScientificReportGenerator örneği al
        try:
            report_generator = get_report_generator()
            logger.info("Scientific Report Generator initialized")
        except Exception as e:
            logger.error(f"Error creating report generator: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Raporları oluştur
        try:
            # CSV raporu oluştur
            logger.info(f"Creating CSV report at: {csv_file}")
            try:
                stats_df = report_generator.gaze_statistics_to_df(gaze_stats)
                
                if stats_df.empty:
                    logger.warning("Generated DataFrame is empty! Check input statistics.")
                else:
                    logger.info(f"DataFrame generated with {len(stats_df)} rows and columns: {stats_df.columns.tolist()}")
                    
                # CSV dosyasını yazmadan önce dizinin var olduğundan emin ol
                if not os.path.exists(os.path.dirname(csv_file)):
                    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
                    logger.info(f"Created directory for CSV file: {os.path.dirname(csv_file)}")
                
                stats_df.to_csv(csv_file, index=False)
                logger.info(f"CSV saved successfully to {csv_file}")
                
                # Dosya gerçekten oluştu mu kontrol et
                if os.path.exists(csv_file):
                    logger.info(f"CSV file exists at: {csv_file}")
                    # Dosya boyutunu kontrol et
                    file_size = os.path.getsize(csv_file)
                    logger.info(f"CSV file size: {file_size} bytes")
                else:
                    logger.error(f"CSV file was not created at: {csv_file}")
            except Exception as csv_err:
                logger.error(f"Error creating CSV report: {csv_err}")
                import traceback
                logger.error(traceback.format_exc())
            
            # PDF raporunu oluştur
            logger.info(f"Creating PDF report at: {pdf_file}")
            try:
                # Önce reportlab'ın gerekli modüllerini kontrol et
                try:
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
                    from reportlab.lib.pagesizes import A4
                    from reportlab.lib import colors
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    logger.info("All required ReportLab modules loaded successfully")
                except ImportError as imp_err:
                    logger.error(f"Error importing ReportLab modules: {imp_err}")
                    raise
                
                # PDF klasörünün var olduğundan emin ol
                if not os.path.exists(os.path.dirname(pdf_file)):
                    os.makedirs(os.path.dirname(pdf_file), exist_ok=True)
                    logger.info(f"Created directory for PDF file: {os.path.dirname(pdf_file)}")
                
                # PDF oluştur
                pdf_path = report_generator.gaze_statistics_report_to_pdf(gaze_stats, pdf_file)
                
                if pdf_path is None:
                    logger.warning("PDF report generation failed - returned None path")
                else:
                    # Dosyanın varlığını kontrol et
                    if os.path.exists(pdf_path):
                        logger.info(f"PDF report created successfully at: {pdf_path}")
                        # Dosya boyutunu kontrol et
                        file_size = os.path.getsize(pdf_path)
                        logger.info(f"PDF file size: {file_size} bytes")
                    else:
                        logger.error(f"PDF report path returned but file doesn't exist: {pdf_path}")
            except Exception as pdf_err:
                logger.error(f"Error creating PDF report: {pdf_err}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Oluşturulan tüm dosyaları bul
            result = {
                "csv": csv_file,
                "pdf": pdf_file if pdf_path is not None else None,
                "directory": {
                    "root": gaze_analysis_dir,
                    "raw_data": gaze_raw_data_dir,
                    "figures": gaze_figures_dir,
                    "tables": gaze_tables_dir
                }
            }
            
            # figures ve tables klasörlerindeki dosyaları ekle
            figure_files = []
            for file in os.listdir(gaze_figures_dir):
                if file.endswith('.png') or file.endswith('.jpg'):
                    figure_files.append(os.path.join(gaze_figures_dir, file))
            
            # Eğer figür dosyaları yoksa, manuel olarak oluştur
            if not figure_files:
                try:
                    # matplotlib'ı import et ve grafikler için ayarla
                    import matplotlib
                    matplotlib.use('Agg')  # GUI olmayan ortamlar için
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    logger.info("No figures found. Creating figures manually...")
                    
                    # zone_percentages'tan pasta grafiği oluştur
                    zone_names = list(gaze_stats["zone_percentages"].keys())
                    percentages = list(gaze_stats["zone_percentages"].values())
                    
                    # Sıfır olmayan değerleri filtrele (0.1'den büyük yüzdeler)
                    filtered_data = [(name, pct) for name, pct in zip(zone_names, percentages) if pct > 0.1]
                    if filtered_data:
                        filtered_zones, filtered_percentages = zip(*filtered_data)
                        
                        # Pasta grafik oluştur
                        plt.figure(figsize=(8, 6))
                        plt.pie(filtered_percentages, labels=filtered_zones, autopct='%1.1f%%', 
                                startangle=90, shadow=True)
                        plt.axis('equal')  # Daire şeklinde
                        plt.title('Gaze Distribution by Zone')
                        
                        # Kaydet
                        pie_file = os.path.join(gaze_figures_dir, f"gaze_pie_{date_str}.png")
                        plt.savefig(pie_file, dpi=300)
                        plt.close()
                        
                        figure_files.append(pie_file)
                        logger.info(f"Created pie chart: {pie_file}")
                    
                    # Zone durations çubuk grafiği
                    plt.figure(figsize=(10, 6))
                    plt.bar(zone_names, list(gaze_stats["zone_durations"].values()))
                    plt.title('Zone Durations')
                    plt.xlabel('Zone')
                    plt.ylabel('Duration (seconds)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    bar_file = os.path.join(gaze_figures_dir, f"zone_durations_bar_{date_str}.png")
                    plt.savefig(bar_file, dpi=300)
                    plt.close()
                    
                    figure_files.append(bar_file)
                    logger.info(f"Created bar chart: {bar_file}")
                except Exception as e:
                    logger.error(f"Error creating manual figures: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            table_files = []
            for file in os.listdir(gaze_tables_dir):
                if file.endswith('.csv') or file.endswith('.xlsx'):
                    table_files.append(os.path.join(gaze_tables_dir, file))
            
            # Eğer tablo dosyaları yoksa, manuel olarak oluştur
            if not table_files:
                try:
                    logger.info("No table files found. Creating tables manually...")
                    
                    # Excel tablosu oluştur
                    import pandas as pd
                    
                    # Veri çerçevesi oluştur
                    data = []
                    for zone, duration in gaze_stats["zone_durations"].items():
                        percentage = gaze_stats["zone_percentages"].get(zone, 0)
                        visits = gaze_stats["zone_visit_counts"].get(zone, 0)
                        
                        # Ortalama süreyi hesapla
                        avg_duration = 0
                        if visits > 0:
                            avg_duration = duration / visits
                        
                        data.append({
                            "Zone": zone,
                            "Duration (s)": round(duration, 2),
                            "Percentage (%)": round(percentage, 2),
                            "Visit Count": visits,
                            "Avg Duration (s)": round(avg_duration, 2)
                        })
                    
                    # DataFrame oluştur
                    if data:
                        df = pd.DataFrame(data)
                        
                        # CSV dosyası oluştur
                        csv_file_table = os.path.join(gaze_tables_dir, f"gaze_stats_{date_str}.csv")
                        df.to_csv(csv_file_table, index=False)
                        table_files.append(csv_file_table)
                        logger.info(f"Created CSV table: {csv_file_table}")
                        
                        # Excel dosyası oluştur
                        excel_file = os.path.join(gaze_tables_dir, f"gaze_stats_{date_str}.xlsx")
                        df.to_excel(excel_file, index=False)
                        table_files.append(excel_file)
                        logger.info(f"Created Excel table: {excel_file}")
                except Exception as e:
                    logger.error(f"Error creating manual tables: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            result["figures"] = figure_files
            result["tables"] = table_files
            
            logger.info(f"Gaze statistics report generated with {len(figure_files)} figures and {len(table_files)} tables")
            
            if not figure_files:
                logger.warning("No figure files were generated - this may indicate a problem!")
            if not table_files:
                logger.warning("No table files were generated - this may indicate a problem!")
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # İstisna oluştuğunda bile CSV'yi döndürmeye çalış
            if os.path.exists(csv_file):
                return {"csv": csv_file}
            return None
            
    except Exception as e:
        logger.error(f"Error in generate_gaze_statistics_report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_distraction_report(report_data: Dict[str, Any], timestamp: float = None) -> Optional[Dict[str, str]]:
    """
    Generate a comprehensive distraction report.
    
    Args:
        report_data: Dictionary containing distraction monitoring data
        timestamp: Optional timestamp for the report (not used, for backward compatibility)
        
    Returns:
        Dict[str, str]: Paths to the generated report files, or None if error
    """
    try:
        # Get or create current session
        session_dir, date_str, _ = get_or_create_current_session()
        
        # Create a session folder for distraction analysis
        distraction_dir = os.path.join(session_dir, 'distraction_analysis')
        os.makedirs(distraction_dir, exist_ok=True)
        
        # Create raw_data subdirectory
        raw_data_dir = os.path.join(distraction_dir, 'raw_data')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Figures subdirectory for distraction analysis
        figures_dir = os.path.join(distraction_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # JSON report path for raw data
        json_path = os.path.join(raw_data_dir, f'distraction_report_{date_str}.json')
        
        # PDF report path in the main distraction_analysis directory, not in raw_data
        pdf_path = os.path.join(distraction_dir, f"distraction_analysis_report_{date_str}.pdf")
        
        # Get report generator
        report_generator = get_report_generator()
        
        # Save JSON report
        report_generator.save_json_report(report_data, json_path)
        logger.info(f"JSON distraction data saved to: {json_path}")
        
        # Generate PDF report using the ScientificReportGenerator
        try:
            # Create PDF document
            from reportlab.lib.pagesizes import A4
            
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=A4,
                rightMargin=72,  # Sabit marj değeri kullan
                leftMargin=72,   # Sabit marj değeri kullan 
                topMargin=72,    # Sabit marj değeri kullan
                bottomMargin=72  # Sabit marj değeri kullan
            )
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            heading2_style = styles['Heading2']
            normal_style = styles['Normal']
            
            # Add scientific description style
            styles.add(ParagraphStyle(
                name='FigureDescription',
                fontName='Helvetica-Oblique',
                fontSize=10,
                alignment=1,
                spaceAfter=12,
                textColor=colors.darkblue
            ))
            
            # Add title
            title = "Driver Distraction Report"
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 12))
            
            # Add date and time
            if 'date' in report_data:
                date_text = f"Report Date: {report_data['date']}"
            else:
                date_text = f"Report Date: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(date_text, normal_style))
            story.append(Spacer(1, 12))
            
            # Add distraction summary
            if 'distraction' in report_data:
                distraction = report_data['distraction']
                story.append(Paragraph("Distraction Summary", heading2_style))
                story.append(Spacer(1, 6))
                
                level_text = f"Distraction Level: {distraction.get('level', 'N/A')}"
                story.append(Paragraph(level_text, normal_style))
                
                score_text = f"Distraction Score: {distraction.get('score', 0):.1f}/100"
                story.append(Paragraph(score_text, normal_style))
                
                if 'details' in distraction and 'reasons' in distraction['details']:
                    story.append(Paragraph("Distraction Reasons:", normal_style))
                    for reason in distraction['details']['reasons']:
                        story.append(Paragraph(f"• {reason}", normal_style))
                
                story.append(Spacer(1, 12))
            
            # Add EU regulation compliance
            if 'eu_regulation' in report_data:
                eu_reg = report_data['eu_regulation']
                story.append(Paragraph("EU Regulation Compliance", heading2_style))
                story.append(Spacer(1, 6))
                
                # Debug için eu_regulation değerlerini loglayalım
                logger.info(f"EU Regulation data for report: {eu_reg}")
                
                compliance_text = f"Overall Compliance: {eu_reg.get('overall_compliance', 'N/A')}"
                story.append(Paragraph(compliance_text, normal_style))

                # Eğer rapora ekstra açıklama eklenmek istenirse:
                if 'area1_statistics' in report_data:
                    area1_stats = report_data['area1_statistics']
                    area1_debug = f"Area 1 Debug - Total Time: {area1_stats.get('total_time', 0):.2f}s, " + \
                                  f"Total Driving Time: {report_data.get('total_driving_time', 0):.2f}s, " + \
                                  f"Percentage: {area1_stats.get('percentage', 0):.2f}%"
                    logger.info(area1_debug)
                
                # Create a table for compliance details
                data = [
                    ["Metric", "Limit", "Actual", "Compliant"],
                    ["Area 1 Percentage", f"{eu_reg.get('area1_percentage_limit', 0):.1f}%", 
                     f"{eu_reg.get('area1_percentage_actual', 0):.1f}%", 
                     "Yes" if eu_reg.get('area1_percentage_compliant', False) else "No"],
                    ["Road Center Absence", f"{eu_reg.get('road_center_absence_limit', 0):.1f}s", 
                     f"{eu_reg.get('road_center_absence_actual', 0):.2f}s", 
                     "Yes" if eu_reg.get('road_center_absence_compliant', False) else "No"],
                    ["Area 1 to Area 2 Transition", "1.0s", 
                     f"{eu_reg.get('area1_to_area2_transition_time', 0):.2f}s", 
                     "Yes" if eu_reg.get('area1_to_area2_compliant', False) else "No"]
                ]
                
                t = Table(data, colWidths=[150, 100, 100, 80])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))
            
            # Add zone statistics if available
            if 'zone_statistics' in report_data:
                story.append(Paragraph("Gaze Zone Statistics", heading2_style))
                story.append(Spacer(1, 12))
                
                zones = report_data['zone_statistics']
                zone_data = [["Zone", "Duration (s)", "Visits", "Percentage (%)"]]
                
                for zone_id, zone_info in zones.items():
                    zone_name = zone_info.get('name', f'Zone {zone_id}')
                    duration = zone_info.get('total_duration', 0)
                    visits = zone_info.get('visit_count', 0)
                    percentage = zone_info.get('percentage', 0)
                    
                    # Yüzde değerlerinin doğru gösterildiğinden emin ol
                    zone_data.append([
                        zone_name,
                        f"{duration:.1f}",
                        str(visits),
                        f"{percentage:.1f}"
                    ])
                
                # Debug için zone bilgilerini logla
                logger.info(f"Zone statistics for PDF report: {zones}")
                
                zone_table = Table(zone_data, colWidths=[150, 100, 100, 100])
                zone_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                ]))
                story.append(zone_table)
            
            # Check for available figures in the figures directory
            available_figures = []
            if os.path.exists(figures_dir):
                for file in os.listdir(figures_dir):
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        available_figures.append(os.path.join(figures_dir, file))
            
            # If no figures found, try to create some
            if not available_figures:
                # Figürlerin oluşturulması özelliğini kaldır, bu fonksiyonu içeriğini değiştir
                logger.info("Skipping figure creation for distraction report as requested")
                # Figür aramaya devam etmeyelim, zaten boş bir liste olacak
                available_figures = []
            
            # Add figures to the report with descriptions
            if available_figures:
                story.append(Paragraph("Distraction Analysis Figures", heading2_style))
                story.append(Spacer(1, 6))
                
                # Scientific descriptions for possible figures
                figure_descriptions = {
                    "pie": """
                    <b>Figure 1: Gaze Distribution Analysis</b><br/>
                    This pie chart visualizes the distribution of the driver's visual attention across different zones in the vehicle. 
                    Scientific analysis of this distribution enables assessment of attentional resource allocation during driving tasks.
                    EU regulation C(2023)4523 specifies that excessive duration (>15%) in non-driving areas (Area 1) 
                    may indicate increased risk of cognitive distraction and decreased situational awareness.
                    """,
                    
                    "bar": """
                    <b>Figure 2: Zone Duration Comparison</b><br/>
                    This chart presents comparative durations spent in each gaze zone, quantifying visual attention patterns.
                    Scientific research indicates that balanced distribution between driving-critical zones correlates with 
                    improved situational awareness and reduced accident risk. Prolonged fixation in any single zone,
                    particularly those not critical for driving (e.g., infotainment), indicates potential attentional tunneling.
                    """,
                    
                    "transition": """
                    <b>Figure 3: Gaze Transition Analysis</b><br/>
                    This visualization quantifies the frequency of gaze transitions between different visual zones.
                    According to cognitive attention theories, transition patterns reveal scanning strategies and attentional 
                    flexibility during complex driving tasks. High transition frequency between driving-critical zones
                    indicates effective environmental monitoring, while limited transitions suggest reduced situational awareness.
                    """
                }
                
                # Add each figure with appropriate description
                for i, fig_path in enumerate(available_figures):
                    # Figure type detection
                    fig_type = None
                    if "pie" in fig_path.lower():
                        fig_type = "pie"
                    elif "bar" in fig_path.lower() and "transition" not in fig_path.lower():
                        fig_type = "bar"
                    elif "transition" in fig_path.lower():
                        fig_type = "transition"
                    
                    # Add figure
                    img = Image(fig_path, width=400, height=300)
                    story.append(img)
                    story.append(Spacer(1, 6))
                    
                    # Add description if available
                    if fig_type and fig_type in figure_descriptions:
                        story.append(Paragraph(figure_descriptions[fig_type], styles['FigureDescription']))
                    else:
                        story.append(Paragraph(f"<b>Figure {i+1}</b>: Analysis of driver's gaze patterns and distraction metrics.", 
                                             styles['FigureDescription']))
                    
                    story.append(Spacer(1, 12))
            
            # Add zone transitions statistics if available
            if 'transitions' in report_data:
                transitions_data = report_data['transitions']
                if 'area1_to_area2_time_avg' in transitions_data:
                    story.append(Paragraph("Gaze Transition Analysis", heading2_style))
                    story.append(Spacer(1, 6))
                    
                    avg_time = transitions_data.get('area1_to_area2_time_avg', 0)
                    transition_count = transitions_data.get('transition_count', 0)
                    
                    transition_text = f"""
                    <b>Transition Metrics:</b><br/>
                    • Average transition time from Area 1 to Area 2: {avg_time:.2f}s<br/>
                    • Total number of transitions: {transition_count}<br/>
                    • Transition frequency: {transitions_data.get('transition_frequency', 0):.2f} transitions/minute<br/><br/>
                    
                    <i>Note: According to scientific literature and EU regulations, transition time from non-driving areas
                    to driving-critical areas should ideally be under 1.0 second to maintain adequate situational awareness.
                    Longer transition times may indicate decreased cognitive readiness to respond to driving events.</i>
                    """
                    
                    story.append(Paragraph(transition_text, normal_style))
                    story.append(Spacer(1, 12))
                    
                    # Add a table for road center returns if available
                    if 'road_center_returns' in transitions_data and transitions_data['road_center_returns']:
                        rc_returns = transitions_data['road_center_returns']
                        if rc_returns:
                            story.append(Paragraph("Road Center Return Times (seconds)", normal_style))
                            
                            # Format the data
                            rc_data = []
                            chunk_size = 5  # Show 5 values per row
                            for i in range(0, len(rc_returns), chunk_size):
                                chunk = rc_returns[i:i+chunk_size]
                                rc_data.append([f"{val:.2f}" for val in chunk])
                            
                            # Create the table
                            if rc_data:
                                rc_table = Table(rc_data)
                                rc_table.setStyle(TableStyle([
                                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                    ('FONTSIZE', (0, 0), (-1, -1), 8)
                                ]))
                                story.append(rc_table)
                                story.append(Spacer(1, 6))
                                
                                # Add scientific context
                                rc_context = """
                                <i>Scientific Context: Return-to-road-center times reflect the driver's ability to reorient 
                                attention to the primary driving task after attending to secondary tasks or other visual zones.
                                Longer return times correlate with increased risk of missed hazards and delayed responses to driving events.</i>
                                """
                                story.append(Paragraph(rc_context, styles['FigureDescription']))
                                story.append(Spacer(1, 12))
            
            # Build the PDF document
            doc.build(story)
            logger.info(f"PDF distraction report saved to: {pdf_path}")
        except ImportError as e:
            logger.warning(f"Could not generate PDF report: {str(e)}")
            logger.warning("PDF generation requires reportlab package")
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return {
            'json': json_path,
            'pdf': pdf_path
        }
    except Exception as e:
        logger.error(f"Error generating distraction report: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None 