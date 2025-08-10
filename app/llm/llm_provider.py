"""
LLM Provider - Handles interactions with Language Model APIs
Optimized for free tiers (Gemini) with rate limiting and fallbacks
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import google.generativeai as genai

# A simple retry mechanism is added to handle potential API issues.
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.utils import get_logger

logger = get_logger(__name__)

class LLMProvider:
    """Manages LLM API interactions with rate limiting for free tiers"""
    
    def __init__(self):
        self.client = None
        # Using a fast, student-friendly model suitable for free-tier use.
        self.model_name = "gemini-1.5-flash"  
        self.generation_config = {
            "temperature": 0.2, # Keep the temperature low for more deterministic code.
            "max_output_tokens": 4096, # Set a generous token limit.
        }
        
        # Rate limiting for free tier (15 requests/minute)
        self.request_history = []
        self.min_request_interval = 4.0  # 4 seconds between requests
        self.max_requests_per_minute = 12  # Conservative limit
        
        # Retry configuration
        self.max_retries = 2
        self.retry_delay = 5.0
        
        # The prompt that instructs the LLM on how to behave.
        # This is crucial for getting the desired output format (Python code).
        self.system_instruction = (
            "You are an expert Python data analyst. Your task is to write a single, "
            "complete, and runnable Python script to answer a user's data analysis questions. "
            "The script should be self-contained and use the provided pandas DataFrame `df`. "
            "Use only the following libraries: pandas, numpy, matplotlib. "
            "For plotting, always save the plot as a base64 encoded PNG and return it as a data URI. "
            "Your final output should be a Python list with four elements: [number, string, float, base64_image]. "
            "The code should not use any external files or non-standard libraries. "
            "Do not include any text or explanations outside of the code block."
        )

    async def initialize(self):
        """Initialize the LLM client"""
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            logger.warning("No GEMINI_API_KEY found - LLM features will be disabled")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.client = genai
            logger.info("Gemini API client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            self.client = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_analysis_code(self, prompt: str) -> Optional[str]:
        """
        Generates Python code based on the provided prompt using the Gemini API.
        
        Args:
            prompt: The full analysis request, including questions and data context.

        Returns:
            A string containing the generated Python code, or None if the call fails.
        """
        if not self.client:
            logger.error("LLM client not initialized. Cannot generate code.")
            return None

        # Check and respect rate limits before making the call.
        await self._wait_for_rate_limit()

        try:
            logger.info("Calling Gemini API to generate analysis code...")
            model = self.client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction
            )
            response = await model.generate_content_async(
                prompt,
                generation_config=self.generation_config
            )
            self.request_history.append(datetime.now())
            
            # The model is expected to return a single block of Python code.
            analysis_code = response.text.strip('`').strip('python').strip()
            
            if not analysis_code:
                raise ValueError("LLM response was empty or did not contain code.")
                
            return analysis_code
        except Exception as e:
            logger.error(f"Error generating analysis code: {e}")
            raise # Re-raise to trigger the retry mechanism.

    async def _wait_for_rate_limit(self):
        """Wait if needed to respect free-tier rate limits"""
        
        # Check requests per minute
        recent_requests = [req for req in self.request_history if (datetime.now() - req).total_seconds() < 60]
        self.request_history = recent_requests
        
        if len(recent_requests) >= self.max_requests_per_minute:
            wait_time = 60 - (datetime.now() - recent_requests[0]).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Check minimum interval
        if self.request_history:
            time_since_last = (datetime.now() - self.request_history[-1]).total_seconds()
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                logger.debug(f"Waiting {wait_time:.1f}s for rate limit")
                await asyncio.sleep(wait_time)

