"""
Data Analyst Agent - Main Application Entry Point
"""

import os
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
import psutil

from app.handler import app, handler
from app.utils.utils import setup_logging, load_config


logger = setup_logging()

config = load_config()

@app.get("/")
async def read_root():
    """Root endpoint for the API"""
    return {"message": "Data Analyst Agent is running!"}

@app.get("/health")
async def health_check():
    """Comprehensive health check for the application"""
    
    llm_status = "ok" if handler.llm_provider.client else "disabled"
    
    try:
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
    except Exception:
        memory_percent = -1

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


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("app.handler:app", host=host, port=port, reload=True)
