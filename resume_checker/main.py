"""Main ResumeChecker class that orchestrates the complete analysis process."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json

from .extractors import TextExtractor
from .similarity import SimilarityCalculator
from .reviewers import AIReviewer
from .utils import Config, TextPreprocessor


class ResumeChecker:
    """
    Main class for AI-powered resume analysis against job descriptions.
    
    This class orchestrates the complete workflow:
    1. Extract text from resume files
    2. Calculate similarity with job descriptions
    3. Generate detailed AI-powered reviews
    """
    
    def __init__(self, config: Optional[Config] = None, groq_api_key: Optional[str] = None):
        """
        Initialize the ResumeChecker.
        
        Args:
            config: Configuration object (optional)
            groq_api_key: Groq API key (optional, overrides config)
        """
        # Setup configuration
        self.config = config or Config()
        
        # Override API key if provided
        if groq_api_key:
            self.config.set('groq_api_key', groq_api_key)
        
        # Setup logging
        self.logger = self.config.setup_logging()
        
        # Initialize components
        self.text_extractor = TextExtractor()
        self.text_preprocessor = TextPreprocessor()
        
        # Initialize similarity calculator
        similarity_config = self.config.get_similarity_config()
        self.similarity_calculator = SimilarityCalculator(
            model_name=similarity_config['model_name'],
            use_tfidf_fallback=True
        )
        
        # Initialize AI reviewer (lazy loading)
        self._ai_reviewer = None
        
        self.logger.info("ResumeChecker initialized successfully")
    
    @property
    def ai_reviewer(self) -> AIReviewer:
        """Lazy load the AI reviewer."""
        if self._ai_reviewer is None:
            groq_config = self.config.get_groq_config()
            if not groq_config['api_key']:
                raise ValueError("Groq API key is required for AI reviews. Set GROQ_API_KEY environment variable.")
            
            self._ai_reviewer = AIReviewer(
                api_key=groq_config['api_key'],
                model=groq_config['model']
            )
        
        return self._ai_reviewer
    
    def analyze_resume(self, resume_path: str, job_description: str, 
                      include_ai_review: bool = True) -> Dict[str, Any]:
        """
        Perform complete resume analysis against a job description.
        
        Args:
            resume_path: Path to the resume file
            job_description: Job description text or file path
            include_ai_review: Whether to include AI-powered review
            
        Returns:
            Complete analysis results
        """
        try:
            self.logger.info(f"Starting analysis for resume: {resume_path}")
            
            # Step 1: Extract resume text
            resume_data = self.extract_resume_text(resume_path)
            resume_text = resume_data['text']
            
            # Handle job description (text or file path)
            if Path(job_description).exists():
                job_data = self.extract_resume_text(job_description)
                job_text = job_data['text']
                self.logger.info("Job description loaded from file")
            else:
                job_text = job_description
                self.logger.info("Job description provided as text")
            
            # Step 2: Preprocess texts
            processed_resume = self._preprocess_text(resume_text)
            processed_job = self._preprocess_text(job_text)
            
            # Step 3: Calculate similarity
            similarity_data = self.calculate_similarity(processed_resume, processed_job)
            
            # Step 4: Generate AI review (if requested)
            ai_review = None
            if include_ai_review:
                try:
                    ai_review = self.generate_ai_review(resume_text, job_text, similarity_data)
                except Exception as e:
                    self.logger.error(f"AI review generation failed: {str(e)}")
                    ai_review = {"error": f"AI review unavailable: {str(e)}"}
            
            # Compile results
            results = {
                "resume_info": {
                    "file_path": resume_path,
                    "extraction_data": resume_data,
                    "processed_length": len(processed_resume)
                },
                "job_description_info": {
                    "source": "file" if Path(job_description).exists() else "text",
                    "processed_length": len(processed_job)
                },
                "similarity_analysis": similarity_data,
                "ai_review": ai_review,
                "analysis_metadata": {
                    "timestamp": self._get_timestamp(),
                    "config_used": {
                        "similarity_model": self.config.get('similarity_model'),
                        "groq_model": self.config.get('groq_model'),
                        "include_ai_review": include_ai_review
                    }
                }
            }
            
            self.logger.info(f"Analysis completed successfully. Match score: {similarity_data.get('similarity_score', 0):.1f}%")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in resume analysis: {str(e)}")
            raise
    
    def extract_resume_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a resume file.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Extracted text and metadata
        """
        return self.text_extractor.extract_text(file_path)
    
    def calculate_similarity(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Calculate similarity between resume and job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Similarity metrics and analysis
        """
        return self.similarity_calculator.calculate_similarity(resume_text, job_description)
    
    def generate_ai_review(self, resume_text: str, job_description: str, 
                          similarity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate AI-powered review of the resume against job description.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            similarity_data: Similarity analysis results
            
        Returns:
            AI-generated review
        """
        return self.ai_reviewer.generate_review(resume_text, job_description, similarity_data)
    
    def quick_score(self, resume_path: str, job_description: str) -> Dict[str, Any]:
        """
        Get a quick compatibility score without full analysis.
        
        Args:
            resume_path: Path to the resume file
            job_description: Job description text or file path
            
        Returns:
            Quick score and basic metrics
        """
        try:
            # Extract texts
            resume_data = self.extract_resume_text(resume_path)
            resume_text = resume_data['text']
            
            if Path(job_description).exists():
                job_data = self.extract_resume_text(job_description)
                job_text = job_data['text']
            else:
                job_text = job_description
            
            # Quick similarity calculation
            processed_resume = self._preprocess_text(resume_text)
            processed_job = self._preprocess_text(job_text)
            similarity_data = self.calculate_similarity(processed_resume, processed_job)
            
            # Quick AI score (if available)
            ai_score = None
            try:
                ai_score = self.ai_reviewer.generate_quick_score(resume_text, job_text)
            except Exception as e:
                self.logger.warning(f"Quick AI score unavailable: {str(e)}")
            
            return {
                "similarity_score": similarity_data.get('similarity_score', 0),
                "match_category": similarity_data.get('match_category', 'Unknown'),
                "ai_score": ai_score,
                "quick_metrics": {
                    "resume_length": len(resume_text),
                    "job_description_length": len(job_text),
                    "method": similarity_data.get('embedding_method', 'unknown')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in quick score calculation: {str(e)}")
            raise
    
    def batch_analyze(self, resume_paths: List[str], job_descriptions: List[str], 
                     include_ai_review: bool = False) -> List[Dict[str, Any]]:
        """
        Analyze multiple resumes against multiple job descriptions.
        
        Args:
            resume_paths: List of resume file paths
            job_descriptions: List of job description texts or file paths
            include_ai_review: Whether to include AI reviews
            
        Returns:
            List of analysis results
        """
        if len(resume_paths) != len(job_descriptions):
            raise ValueError("Number of resumes must match number of job descriptions")
        
        results = []
        total = len(resume_paths)
        
        for i, (resume_path, job_desc) in enumerate(zip(resume_paths, job_descriptions)):
            try:
                self.logger.info(f"Processing batch item {i+1}/{total}")
                result = self.analyze_resume(resume_path, job_desc, include_ai_review)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing batch item {i+1}: {str(e)}")
                results.append({
                    "error": str(e),
                    "resume_path": resume_path,
                    "job_description": job_desc[:100] + "..." if len(job_desc) > 100 else job_desc
                })
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save analysis results to a file.
        
        Args:
            results: Analysis results
            output_path: Path to save the results
        """
        try:
            output_path = Path(output_path)
            
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                # Save as formatted text
                formatted_output = self._format_results_as_text(results)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
            
            self.logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        config = self.config.get_preprocessing_config()
        return self.text_preprocessor.normalize_text_for_similarity(text)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        import datetime
        return datetime.datetime.now().isoformat()
    
    def _format_results_as_text(self, results: Dict[str, Any]) -> str:
        """Format results as readable text."""
        output = []
        
        output.append("=== AI RESUME CHECKER ANALYSIS ===\n")
        
        # Basic info
        if 'similarity_analysis' in results:
            sim = results['similarity_analysis']
            output.append(f"MATCH SCORE: {sim.get('similarity_score', 0):.1f}%")
            output.append(f"MATCH CATEGORY: {sim.get('match_category', 'Unknown')}")
            output.append("")
        
        # AI Review
        if 'ai_review' in results and results['ai_review'] and 'error' not in results['ai_review']:
            review = results['ai_review']
            
            output.append("OVERALL ASSESSMENT:")
            output.append(review.get('overall_assessment', 'Not available'))
            output.append("")
            
            output.append("STRENGTHS:")
            for i, strength in enumerate(review.get('strengths', []), 1):
                output.append(f"{i}. {strength}")
            output.append("")
            
            output.append("AREAS FOR IMPROVEMENT:")
            for i, weakness in enumerate(review.get('weaknesses', []), 1):
                output.append(f"{i}. {weakness}")
            output.append("")
            
            output.append("IMPROVEMENT SUGGESTIONS:")
            for i, suggestion in enumerate(review.get('improvement_suggestions', []), 1):
                output.append(f"{i}. {suggestion}")
            output.append("")
        
        # Technical details
        if 'analysis_metadata' in results:
            output.append("TECHNICAL DETAILS:")
            metadata = results['analysis_metadata']
            output.append(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}")
            if 'config_used' in metadata:
                config = metadata['config_used']
                output.append(f"Similarity Model: {config.get('similarity_model', 'Unknown')}")
                output.append(f"AI Model: {config.get('groq_model', 'Unknown')}")
        
        return "\n".join(output)