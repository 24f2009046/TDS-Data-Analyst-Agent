"""
Request Handler - Main business logic for data analysis requests
"""

import asyncio
import io
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.llm.llm_provider import LLMProvider
from app.analysis.data_scraper import DataScraper
from app.analysis.data_analyzer import DataAnalyzer
from app.utils.utils import get_logger, timer


logger = get_logger(__name__)


class DataAnalysisHandler:
    """Main handler for data analysis requests"""
    
    def __init__(self):
        self.llm_provider = LLMProvider()
        self.data_scraper = DataScraper()
        self.data_analyzer = DataAnalyzer(self.llm_provider)
        
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "llm_calls": 0,
            "scraping_requests": 0,
            "fallback_used": 0
        }
        
        self.cache = {}
        self.initialized = False
        
    async def initialize(self):
        if self.initialized:
            return
        
        logger.info("Initializing application components...")
        
        await self.llm_provider.initialize()
        await self.data_scraper.initialize()
        
        self.initialized = True
        
    async def cleanup(self):
        logger.info("Cleaning up application components...")
        await self.data_scraper.cleanup()
        
    @timer
    async def handle_request(self, files: List[UploadFile]) -> List[Any]:
        self.metrics["total_requests"] += 1
        
        try:
            questions_file = next((f for f in files if f.filename == "questions.txt"), None)
            if not questions_file:
                raise HTTPException(status_code=400, detail="questions.txt is required.")
            
            questions = (await questions_file.read()).decode('utf-8')
            
            # Find the primary data file (e.g., CSV) and other files
            main_data_file = next((f for f in files if f.filename and f.filename.endswith(('.csv'))), None)
            additional_files = {f.filename: await f.read() for f in files if f.filename not in ["questions.txt", main_data_file.filename if main_data_file else None]}
            
            df = None
            if main_data_file:
                df = pd.read_csv(io.StringIO((await main_data_file.read()).decode('utf-8')))
            else:
                logger.warning("No CSV data file provided. Analysis will be based on context and LLM-generated data.")

            result = await self.data_analyzer.analyze_with_llm(questions, df, additional_files)
            
            self.metrics["successful_requests"] += 1
            return result
        
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self.metrics["failed_requests"] += 1
            # Fallback for severe errors or if the LLM provider is disabled
            return await self.data_analyzer._fallback_analysis(str(e), df if 'df' in locals() else None)
            
    def get_metrics(self) -> Dict[str, Any]:
        success_rate = (
            self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1) * 100
        )
        
        return {
            "total_requests": self.metrics["total_requests"],
            "success_rate_percent": round(success_rate, 2),
            "llm_calls": self.metrics["llm_calls"],
            "scraping_requests": self.metrics["scraping_requests"],
            "fallback_used": self.metrics["fallback_used"]
        }


handler = DataAnalysisHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Data Analyst Agent...")
    await handler.initialize()
    logger.info("Data Analyst Agent started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Data Analyst Agent...")
    await handler.cleanup()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Data Analyst Agent - Student Edition",
    version="1.0.0",
    description="AI-powered data analysis API optimized for student projects and free tiers",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Main API endpoint for data analysis
    """
    return await handler.handle_request(files)
