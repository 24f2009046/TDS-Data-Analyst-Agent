"""
Data Analyst Agent - Main Application Entry Point
Student Edition - Optimized for free API tiers and 3-minute response limits
"""

import os
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psutil

# Import the main application instance and handler from your handler.py module.
# This ensures a modular design where all API logic is defined in one place.
from app.handler import app, handler
from app.utils.utils import setup_logging, load_config


# Initialize logging
logger = setup_logging()

# Load configuration
config = load_config()

# Add CORS middleware to the main app instance.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    """Root endpoint for the API"""
    return {"message": "Data Analyst Agent is running!"}

@app.get("/health")
async def health_check():
    """Comprehensive health check for the application"""
    
    # Check if LLM provider is initialized
    llm_status = "ok" if handler.llm_provider.client else "disabled"
    
    # Get system metrics
    try:
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
    except Exception:
        memory_percent = -1 # Indicate an error in fetching memory

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": llm_status,
        "system": {
            "memory_usage_percent": memory_percent,
            "python_version": config.get("python_version", "unknown")
        },
        "config": {
            "max_request_timeout": config.get("max_request_timeout", 180),
            "scraping_timeout": config.get("scraping_timeout", 30),
            "llm_timeout": config.get("llm_timeout", 45)
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Simple metrics endpoint for monitoring"""
    return handler.get_metrics()


# The startup and shutdown events are already defined in handler.py and attached to the 'app' instance,
# so they will automatically run when this main file starts and stops the application.

if __name__ == "__main__":
    # For local development
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting uvicorn on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
