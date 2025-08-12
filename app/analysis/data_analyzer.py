"""
Data Analyzer - Performs data analysis using LLM and fallback methods
"""

import json
import base64
import io
import re
import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

from app.llm.llm_provider import LLMProvider
from app.analysis.data_scraper import DataScraper
from app.utils.utils import get_logger


logger = get_logger(__name__)

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

class DataAnalyzer:
    """Performs intelligent data analysis with LLM assistance and fallbacks"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.data_scraper = DataScraper()
        
        # Configure matplotlib for server use
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['axes.unicode_minus'] = False
    
    async def analyze_with_llm(self, questions: str, df: Optional[pd.DataFrame], additional_files: Optional[Dict[str, bytes]] = None) -> List[Any]:
        """
        Perform analysis using LLM-generated code
        
        Args:
            questions: The analysis questions/tasks.
            df: The main pandas DataFrame to analyze.
            additional_files: A dictionary of other files (e.g., images).

        Returns:
            A list containing the analysis results in the specified format.
        """
        
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(questions, df, additional_files)
            
            # Get code from LLM
            analysis_code = await self.llm_provider.generate_analysis_code(prompt)
            
            if not analysis_code:
                logger.error("LLM failed to generate analysis code. Using fallback.")
                return await self._fallback_analysis(questions, df)
            
            # Execute the generated code
            return await self._execute_analysis_code(analysis_code, df, additional_files, questions)
        
        except Exception as e:
            logger.error(f"Error in analyze_with_llm: {e}")
            return await self._fallback_analysis(questions, df)

    def _create_analysis_prompt(self, questions: str, df: Optional[pd.DataFrame], additional_files: Optional[Dict[str, bytes]]) -> str:
        """
        Create a detailed prompt for the LLM to generate analysis code.
        The prompt includes the questions, the DataFrame's schema, and a few sample rows.
        """
        
        file_info = ""
        if additional_files:
            file_info = "\n\nAdditional files provided:\n"
            for filename in additional_files.keys():
                file_info += f"- {filename}\n"
        
        if df is not None and not df.empty:
            df_head_str = df.head(3).to_string()
            df_info_buffer = io.StringIO()
            df.info(buf=df_info_buffer)
            df_info_str = df_info_buffer.getvalue()
        else:
            df_head_str = "No data provided"
            df_info_str = "No DataFrame info available. You may need to use the `scrape_url` function."
        
        prompt = (
            f"The user wants you to analyze a pandas DataFrame named `df` with the following properties:\n"
            f"DataFrame info:\n{df_info_str}\n\n"
            f"First 3 rows:\n{df_head_str}\n\n"
            f"The questions to answer are:\n{questions}\n"
            f"Your code must return a Python list `final_answer = [number, string, float, base64_image]`. "
            f"For the image, use `matplotlib` to create a plot and the helper function `_save_plot_to_base64()` to convert it.\n\n"
            f"You have access to the following libraries and functions:\n"
            f"- `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `io`, `base64`\n"
            f"- `sklearn.linear_model.LinearRegression`\n"
            f"- `scipy.stats.pearsonr`\n"
            f"- `scrape_url(url)` to scrape a URL into a DataFrame if needed.\n"
            f"Your code must be a single, self-contained Python script. "
            f"Do not use any external files or non-standard libraries.\n\n"
            f"```python\n"
            f"# The helper function _save_plot_to_base64() is already available\n"
            f"# Start your analysis code below. The final answer should be in a list called 'final_answer'.\n"
            f"# Example: final_answer = [42, 'Analysis complete', 3.14, _save_plot_to_base64()]\n"
        )
        
        return prompt

    async def _execute_analysis_code(self, analysis_code: str, df: Optional[pd.DataFrame], additional_files: Optional[Dict[str, bytes]], questions: str) -> List[Any]:
        """
        Safely execute the LLM-generated Python code.
        """
        
        def _save_plot_to_base64() -> str:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            img_data = buffer.getvalue()

            if len(img_data) > 99000:
                logger.warning("Plot image size exceeds 100KB. Re-generating with lower DPI.")
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, "Plot could not be generated at high quality due to size limits.",
                         ha='center', va='center', fontsize=10)
                plt.axis('off')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=50)
                plt.close()
                img_data = buffer.getvalue()

            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        local_scope = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'io': io,
            'base64': base64,
            'LinearRegression': LinearRegression,
            'pearsonr': stats.pearsonr,
            'df': df,
            'additional_files': additional_files,
            '_save_plot_to_base64': _save_plot_to_base64,
            'scrape_url': self.data_scraper.scrape_url,
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'float': float,
                'int': int,
                'list': list,
                'dict': dict,
                'Exception': Exception,
                'isinstance': isinstance,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round,
                '__import__': __import__
            },
        }
        
        try:
            exec(analysis_code, {"__builtins__": {}}, local_scope)
            
            if 'final_answer' in local_scope:
                result = local_scope['final_answer']
                if isinstance(result, list) and len(result) == 4:
                    return result
                else:
                    logger.warning(f"LLM result format invalid: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                    return await self._fallback_analysis(f"Invalid result format: {result}", df)
            else:
                logger.error("Generated code did not produce a 'final_answer' variable.")
                return await self._fallback_analysis("No final_answer variable created", df)

        except Exception as e:
            logger.error(f"Error executing LLM-generated code: {e}")
            return await self._fallback_analysis(f"Code execution error: {str(e)}", df)

    async def _fallback_analysis(self, questions: str, df: Optional[pd.DataFrame]) -> List[Any]:
        """
        Perform a basic fallback analysis if the LLM or generated code fails.
        This provides a safe, non-crashing response.
        """
        logger.warning(f"Using fallback analysis method for: {questions}")
        
        try:
            if df is not None and not df.empty:
                number_answer = len(df)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    float_answer = float(df[numeric_cols[0]].mean())
                    text_answer = f"Fallback analysis: Dataset has {number_answer} rows and {len(df.columns)} columns. Average of '{numeric_cols[0]}': {float_answer:.2f}"
                else:
                    float_answer = 0.0
                    text_answer = f"Fallback analysis: Dataset has {number_answer} rows and {len(df.columns)} columns. No numeric columns found."
                
                plt.figure(figsize=(8, 6))
                if len(numeric_cols) >= 2:
                    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel(numeric_cols[1])
                    plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
                elif len(numeric_cols) == 1:
                    df[numeric_cols[0]].hist(bins=20)
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of {numeric_cols[0]}')
                else:
                    col_counts = [len(df[col].dropna()) for col in df.columns[:5]]
                    plt.bar(range(len(col_counts)), col_counts)
                    plt.xlabel('Column Index')
                    plt.ylabel('Non-null Count')
                    plt.title('Non-null Counts by Column')
                    
            else:
                number_answer = 0
                text_answer = "Fallback analysis: No data provided or data is empty."
                float_answer = 0.0
                
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=14)
                plt.title('Fallback Analysis')
                plt.axis('off')

            plot_base64 = self._save_plot_to_base64_fallback()
            
            return [number_answer, text_answer, float_answer, plot_base64]
            
        except Exception as e:
            logger.error(f"Even fallback analysis failed: {e}")
            return [0, f"Complete analysis failure: {str(e)}", 0.0, self._create_error_plot()]
    
    def _save_plot_to_base64_fallback(self) -> str:
        """
        Internal helper function to save a plot to a base64 string, specifically for fallback.
        """
        try:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            img_data = buffer.getvalue()

            if len(img_data) > 99000:
                logger.warning("Fallback plot image size exceeds 100KB. Re-generating with lower DPI.")
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, "Plot size exceeded limit.", ha='center', va='center')
                plt.axis('off')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=50)
                plt.close()
                img_data = buffer.getvalue()

            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            return self._create_error_plot()
    
    def _create_error_plot(self) -> str:
        """Create a minimal error plot when everything else fails"""
        try:
            plt.figure(figsize=(4, 3))
            plt.text(0.5, 0.5, 'Error\nGenerating\nPlot', ha='center', va='center', fontsize=12)
            plt.axis('off')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=50)
            plt.close()
            img_data = buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
        except:
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    def get_fallback_response(self, error) -> List[Any]:
        """
        Compatibility method for the handler to call fallback analysis
        """
        import asyncio
        
        try:
            loop = asyncio.get_running_loop()
            return [0, f"Error occurred: {str(error)}", 0.0, self._create_error_plot()]
        except RuntimeError:
            return asyncio.run(self._fallback_analysis(str(error), None))
