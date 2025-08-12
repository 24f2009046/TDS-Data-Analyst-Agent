"""
Data Analyzer - Performs data analysis using LLM and fallback methods
"""

import json
import base64
import io
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

# Set matplotlib backend for server environments (no GUI)
plt.switch_backend('Agg')


class DataAnalyzer:
    """Performs intelligent data analysis with LLM assistance and fallbacks"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.data_scraper = DataScraper()

        # Configure matplotlib for server use
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['axes.unicode_minus'] = False

    async def analyze_with_llm(
        self,
        questions: str,
        df: Optional[pd.DataFrame],
        additional_files: Optional[Dict[str, bytes]] = None
    ) -> List[Any]:
        """Perform analysis using LLM-generated code"""
        try:
            prompt = self._create_analysis_prompt(questions, df, additional_files)
            analysis_code = await self.llm_provider.generate_analysis_code(prompt)

            if not analysis_code:
                logger.error("LLM failed to generate analysis code. Using fallback.")
                return await self._fallback_analysis(questions, df)

            return await self._execute_analysis_code(
                analysis_code, df, additional_files, questions
            )

        except Exception as e:
            logger.error(f"Error in analyze_with_llm: {e}")
            return await self._fallback_analysis(questions, df)

    def _create_analysis_prompt(
        self,
        questions: str,
        df: Optional[pd.DataFrame],
        additional_files: Optional[Dict[str, bytes]]
    ) -> str:
        """Create a detailed prompt for the LLM to generate analysis code"""
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
            df_info_str = (
                "No DataFrame info available. "
                "You may need to use the `scrape_url` function."
            )

        prompt = (
            f"The user wants you to analyze a pandas DataFrame named `df`:\n"
            f"DataFrame info:\n{df_info_str}\n\n"
            f"First 3 rows:\n{df_head_str}\n\n"
            f"The questions to answer are:\n{questions}\n"
            f"Your code must return a Python list `final_answer = [number, string, float, base64_image]`.\n"
            f"For the image, use matplotlib to create a plot and `_save_plot_to_base64()` to convert it.\n\n"
            f"You have access to:\n"
            f"- pandas as pd\n"
            f"- numpy as np\n"
            f"- matplotlib.pyplot as plt\n"
            f"- sklearn.linear_model.LinearRegression\n"
            f"- scipy.stats.pearsonr (please use safe_pearsonr(x, y) instead)\n"
            f"- scrape_url(url) → returns a pandas DataFrame synchronously.\n"
            f"⚠ DO NOT import pandas, numpy, matplotlib, sklearn, or scipy — they are already provided.\n"
            f"⚠ DO NOT redefine `pd`, `np`, `plt`, or other provided variables.\n"
            f"Do not use any external files or non-standard libraries.\n\n"
            f"Use `safe_pearsonr(x, y)` instead of `pearsonr(x, y)` for correlation calculations.\n\n"
            f"# final_answer example:\n"
            f"# final_answer = [42, 'Analysis complete', 3.14, _save_plot_to_base64()]\n"
        )

        return prompt

    async def _execute_analysis_code(
        self,
        analysis_code: str,
        df: Optional[pd.DataFrame],
        additional_files: Optional[Dict[str, bytes]],
        questions: str
    ) -> List[Any]:
        """Safely execute the LLM-generated Python code"""

        def _save_plot_to_base64() -> str:
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            img_data = buffer.getvalue()

            if len(img_data) > 99000:
                logger.warning(
                    "Plot image size exceeds 100KB. Regenerating at lower DPI."
                )
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, "Plot too large.", ha="center", va="center")
                plt.axis("off")
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", bbox_inches="tight", dpi=50)
                plt.close()
                img_data = buffer.getvalue()

            return f"data:image/png;base64,{base64.b64encode(img_data).decode('utf-8')}"

        # Sync wrapper for async scrape_url
        def _scrape_url_sync(url: str):
            import nest_asyncio

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(self.data_scraper.scrape_url(url))
            else:
                return asyncio.run(self.data_scraper.scrape_url(url))

        def safe_pearsonr(x, y):
            try:
                res = stats.pearsonr(x, y)
                if isinstance(res, tuple) and len(res) == 2:
                    return res
                else:
                    # In case res is a scalar float64 or other
                    return (res, None)
            except Exception:
                return (np.nan, np.nan)

        safe_builtins = {
            "print": print,
            "len": len,
            "str": str,
            "float": float,
            "int": int,
            "list": list,
            "dict": dict,
            "Exception": Exception,
            "isinstance": isinstance,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
            "__import__": __import__,
        }

        local_scope = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "io": io,
            "BytesIO": io.BytesIO,
            "base64": base64,
            "LinearRegression": LinearRegression,
            "pearsonr": stats.pearsonr,
            "safe_pearsonr": safe_pearsonr,
            "df": df,
            "additional_files": additional_files,
            "_save_plot_to_base64": _save_plot_to_base64,
            "scrape_url": _scrape_url_sync,
        }

        global_scope = {
            "__builtins__": safe_builtins,
            "pd": pd,
            "np": np,
            "plt": plt,
            "io": io,
            "BytesIO": io.BytesIO,
            "base64": base64,
            "LinearRegression": LinearRegression,
            "pearsonr": stats.pearsonr,
            "safe_pearsonr": safe_pearsonr,
            "scrape_url": _scrape_url_sync,
            "_save_plot_to_base64": _save_plot_to_base64,
            "df": df,
            "additional_files": additional_files,
        }

        try:
            exec(analysis_code, global_scope, local_scope)
            if "final_answer" in local_scope:
                result = local_scope["final_answer"]
                if isinstance(result, list) and len(result) == 4:
                    return result
                else:
                    logger.warning(f"Invalid LLM result format: {result}")
                    return await self._fallback_analysis(f"Invalid result: {result}", df)
            else:
                logger.error("No 'final_answer' from LLM code.")
                return await self._fallback_analysis("No final_answer variable", df)

        except Exception as e:
            logger.error(f"Error executing LLM-generated code: {e}")
            return await self._fallback_analysis(f"Execution error: {str(e)}", df)

    async def _fallback_analysis(
        self, questions: str, df: Optional[pd.DataFrame]
    ) -> List[Any]:
        """Basic fallback analysis"""
        logger.warning(f"Using fallback analysis for: {questions}")
        try:
            if df is not None and not df.empty:
                number_answer = len(df)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if numeric_cols.any():
                    float_answer = float(df[numeric_cols[0]].mean())
                    text_answer = (
                        f"Fallback: {number_answer} rows, {len(df.columns)} cols, "
                        f"avg {numeric_cols[0]}: {float_answer:.2f}"
                    )
                else:
                    float_answer = 0.0
                    text_answer = (
                        f"Fallback: {number_answer} rows, {len(df.columns)} cols, "
                        f"no numeric data."
                    )

                plt.figure(figsize=(8, 6))
                if len(numeric_cols) >= 2:
                    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                elif len(numeric_cols) == 1:
                    df[numeric_cols[0]].hist(bins=20)
                else:
                    plt.text(0.5, 0.5, "No numeric columns", ha="center")
            else:
                number_answer = 0
                text_answer = "Fallback: No data provided."
                float_answer = 0.0
                plt.text(0.5, 0.5, "No Data Available", ha="center")

            plot_base64 = self._save_plot_to_base64_fallback()
            return [number_answer, text_answer, float_answer, plot_base64]
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            return [0, f"Analysis failure: {str(e)}", 0.0, self._create_error_plot()]

    def _save_plot_to_base64_fallback(self) -> str:
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        plt.close()
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    def _create_error_plot(self) -> str:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "Error", ha="center", va="center")
        plt.axis("off")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=50)
        plt.close()
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

    def get_fallback_response(self, error) -> List[Any]:
        try:
            loop = asyncio.get_running_loop()
            return [0, f"Error: {str(error)}", 0.0, self._create_error_plot()]
        except RuntimeError:
            return asyncio.run(self._fallback_analysis(str(error), None))
