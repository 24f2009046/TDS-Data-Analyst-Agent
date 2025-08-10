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
from app.utils.utils import get_logger


logger = get_logger(__name__)

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

class DataAnalyzer:
    """Performs intelligent data analysis with LLM assistance and fallbacks"""
    
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        
        # Configure matplotlib for server use
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['axes.unicode_minus'] = False
    
    async def analyze_with_llm(self, questions: str, df: pd.DataFrame, additional_files: Dict[str, bytes] = None) -> List[Any]:
        """
        Perform analysis using LLM-generated code
        
        Args:
            questions: The analysis questions/tasks.
            df: The main pandas DataFrame to analyze.
            additional_files: A dictionary of other files (e.g., images).

        Returns:
            A list containing the analysis results in the specified format.
        """
        
        # Create analysis prompt
        prompt = await self._create_analysis_prompt(questions, df, additional_files)
        
        # Get code from LLM
        analysis_code = await self.llm_provider.generate_analysis_code(prompt)
        
        if not analysis_code:
            logger.error("LLM failed to generate analysis code. Using fallback.")
            return await self._fallback_analysis(questions, df)
        
        # Execute the generated code
        return await self._execute_analysis_code(analysis_code, df, additional_files)

    async def _create_analysis_prompt(self, questions: str, df: pd.DataFrame, additional_files: Dict[str, bytes]) -> str:
        """
        Create a detailed prompt for the LLM to generate analysis code.
        The prompt includes the questions, the DataFrame's schema, and a few sample rows.
        """
        
        # This will contain information about the attached files to give to the LLM.
        file_info = ""
        if additional_files:
            file_info = "\n\nAdditional files provided:\n"
            for filename in additional_files.keys():
                file_info += f"- {filename}\n"
        
        # Give the LLM the structure of the data it will be working with.
        df_head_str = df.head(3).to_string()
        df_info_str = str(df.info())
        
        prompt = (
            f"The user wants you to analyze a pandas DataFrame named `df` with the following properties:\n"
            f"DataFrame info:\n{df_info_str}\n\n"
            f"First 3 rows:\n{df_head_str}\n\n"
            f"The questions to answer are:\n{questions}\n"
            f"Your code must return a Python list `[number, string, float, base64_image]`. "
            f"For the image, use `matplotlib` to create a plot and a helper function `_save_plot_to_base64` to convert it.\n\n"
            f"```python\n"
            f"import pandas as pd\n"
            f"import numpy as np\n"
            f"import matplotlib.pyplot as plt\n"
            f"import io\n"
            f"import base64\n"
            f"from sklearn.linear_model import LinearRegression\n"
            f"from scipy.stats import pearsonr\n\n"
            f"# A helper function to save and encode a plot for the final output\n"
            f"def _save_plot_to_base64() -> str:\n"
            f"    buffer = io.BytesIO()\n"
            f"    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)\n"
            f"    plt.close() # Close the figure to free up memory\n"
            f"    img_data = buffer.getvalue()\n"
            f"    img_base64 = base64.b64encode(img_data).decode('utf-8')\n"
            f"    return f'data:image/png;base64,{img_base64}'\n"
            f"\n"
            f"# Start your analysis code below. The final answer should be in a list called 'final_answer'.\n"
        )
        
        return prompt

    async def _execute_analysis_code(self, analysis_code: str, df: pd.DataFrame, additional_files: Dict[str, bytes]) -> List[Any]:
        """
        Safely execute the LLM-generated Python code.
        
        Args:
            analysis_code: The code string to execute.
            df: The DataFrame to provide to the code's execution environment.
            additional_files: Other files to provide.
        """
        # We create a restricted local namespace for the executed code.
        # This prevents the code from accessing the full environment.
        local_scope = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'io': io,
            'base64': base64,
            'LinearRegression': LinearRegression,
            'pearsonr': pearsonr,
            'df': df,
            'additional_files': additional_files,
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'float': float,
                'int': int,
                'list': list,
                'dict': dict,
                'Exception': Exception,
                'isinstance': isinstance
            },
        }

        # The LLM's code is expected to define a helper function `_save_plot_to_base64`.
        # We need to make this function available in the local scope for the code to call it.
        def _save_plot_to_base64() -> str:
            buffer = io.BytesIO()
            # Set a high DPI for better quality, but check file size.
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            img_data = buffer.getvalue()

            # Ensure the image size is under the 100KB limit
            if len(img_data) > 99000:
                logger.warning("Plot image size exceeds 100KB. Re-generating with lower DPI.")
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, "Plot could not be generated at high quality due to size limits.", ha='center', va='center')
                plt.axis('off')
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=50)
                plt.close()
                img_data = buffer.getvalue()

            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"

        local_scope['_save_plot_to_base64'] = _save_plot_to_base64
        
        try:
            # We use `exec` to run the code. It's a powerful and dangerous function,
            # which is why we are carefully controlling the `local_scope`.
            exec(analysis_code, {"__builtins__": {}}, local_scope)
            
            # The code should have created a variable named `final_answer`.
            if 'final_answer' in local_scope:
                return local_scope['final_answer']
            else:
                raise ValueError("Generated code did not produce a 'final_answer' variable.")

        except Exception as e:
            logger.error(f"Error executing LLM-generated code: {e}")
            return await self._fallback_analysis(questions, df)

    async def _fallback_analysis(self, questions: str, df: pd.DataFrame) -> List[Any]:
        """
        Perform a basic fallback analysis if the LLM or generated code fails.
        This provides a safe, non-crashing response.
        """
        logger.warning("Using fallback analysis method.")
        
        # Simple answers for a safe response.
        number_answer = 0
        text_answer = "Fallback: Could not perform full analysis."
        float_answer = 0.0

        # Create a minimal placeholder plot
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'Analysis Failed', ha='center', va='center', fontsize=14)
        plt.title('Fallback Plot')
        plt.axis('off')
        
        # The same helper function is used for both success and fallback scenarios.
        plot_base64 = self._save_plot_to_base64()
        
        return [number_answer, text_answer, float_answer, plot_base64]
    
    def _save_plot_to_base64(self) -> str:
        """
        Internal helper function to save a plot to a base64 string.
        This is a copy of the function injected into the LLM's code execution environment.
        """
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        img_data = buffer.getvalue()

        # Ensure the image size is under the 100KB limit
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
