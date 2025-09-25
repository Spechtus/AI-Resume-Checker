"""
AI Resume Checker - An AI-powered system for analyzing resumes against job descriptions.

This package provides functionality to:
1. Extract text from various resume formats (PDF, DOCX, TXT)
2. Compute similarity between resumes and job descriptions
3. Generate detailed reviews with strengths, weaknesses, and suggestions
"""

__version__ = "1.0.0"
__author__ = "AI Resume Checker Team"

from .main import ResumeChecker

__all__ = ["ResumeChecker"]