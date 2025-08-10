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

from app.llm.llm_provider import LLMProvider
from app.analysis.data_scraper import DataScraper
from app.analysis.data_analyzer import DataAnalyzer
from app.utils.utils import get_logger, timer


logger = get_logger(__name__)

# Create the FastAPI app instance here, which is standard practice.
# The main.py will then import this instance.
app = FastAPI(
    title="Data Analyst Agent - Student Edition",
    version="1.0.0",
    description="AI-powered data analysis API optimized for student projects and free tiers"
)

class DataAnalysisHandler:
    """Main handler for data analysis requests"""
    
    def __init__(self):
        # We initialize all the core components here.
        self.llm_provider = LLMProvider()
        self.data_scraper = DataScraper()
        self.data_analyzer = DataAnalyzer(self.llm_provider)
        
        # Simple metrics tracking
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "llm_calls": 0,
            "scraping_requests": 0,
            "fallback_used": 0
        }
        
        # In-memory cache for development
        self.cache = {}
    
    async def initialize(self):
        """Initialize handler components"""
        logger.info("Initializing DataAnalysisHandler")
        # Call the initialization methods of the sub-components.
        await self.llm_provider.initialize()
        await self.data_scraper.initialize()
        logger.info("Handler initialized successfully")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")
        # Call cleanup methods.
        await self.data_scraper.cleanup()
        logger.info("Cleanup complete.")
    
    async def handle_request(self, files: List[UploadFile]) -> List[Any]:
        """
        Main logic for processing a data analysis request.
        
        Args:
            files: A list of UploadFile objects from the POST request.

        Returns:
            A list of analysis results.
        """
        self.metrics["total_requests"] += 1
        start_time = datetime.now()
        
        try:
            questions = ""
            df = pd.DataFrame()
            additional_files = {}
            source_url = None

            # Step 1: Read and categorize all uploaded files
            for file in files:
                contents = await file.read()
                filename = file.filename.lower()
                
                if filename == "questions.txt":
                    questions = contents.decode('utf-8')
                    # Check for a URL in the questions.txt
                    url_match = re.search(r'https?://[^\s]+', questions)
                    if url_match:
                        source_url = url_match.group(0)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(contents))
                else:
                    additional_files[file.filename] = contents

            if not questions:
                raise HTTPException(status_code=400, detail="questions.txt file is missing.")
            
            # Step 2: Source the data based on the request
            # Prioritize attached CSV file over scraping from a URL.
            if not df.empty:
                logger.info("Using attached CSV file for analysis.")
            elif source_url:
                logger.info(f"Scraping data from URL: {source_url}")
                self.metrics["scraping_requests"] += 1
                df = await self.data_scraper.scrape_url(source_url)
            else:
                # If no data source is provided, we can use the fallback.
                logger.warning("No CSV or URL provided. Proceeding with fallback.")
                return await self.data_analyzer._fallback_analysis(questions, df)

            if df.empty:
                raise ValueError("Could not source any data for analysis.")

            # Step 3: Run the analysis using the data and questions
            results = await self.data_analyzer.analyze_with_llm(questions, df, additional_files)
            
            self.metrics["successful_requests"] += 1
            return results

        except Exception as e:
            logger.error(f"Request failed: {e}")
            self.metrics["failed_requests"] += 1
            # Return a generic failure response to avoid crashing the server.
            return await self.data_analyzer._fallback_analysis(questions if 'questions' in locals() else "", pd.DataFrame())
        
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            self.metrics["avg_response_time"] = (self.metrics["avg_response_time"] * (self.metrics["total_requests"] - 1) + duration) / self.metrics["total_requests"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        
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

# This is an instance of your handler, which the main application will use.
handler = DataAnalysisHandler()

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Main API endpoint for data analysis
    
    Accepts:
    - questions.txt (required): Analysis questions/tasks
    - Additional files: CSV data, images, etc.
    
    Returns:
    - JSON array with 4 elements: [number, string, float, base64_image]
    """
    return await handler.handle_request(files)

@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("Starting Data Analyst Agent...")
    await handler.initialize()
    logger.info("Data Analyst Agent started successfully")

@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Data Analyst Agent...")
    await handler.cleanup()
    logger.info("Shutdown complete")
