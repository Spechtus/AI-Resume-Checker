"""AI-powered resume review generation using Groq API."""

import logging
from typing import Dict, List, Any, Optional
import json

# Try to import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class AIReviewer:
    """Generates detailed resume reviews using AI."""
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        """
        Initialize the AI reviewer.
        
        Args:
            api_key: Groq API key
            model: Model name to use
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.model = model
        
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not available. Install with: pip install groq")
        
        if not api_key:
            raise ValueError("Groq API key is required")
        
        try:
            self.client = Groq(api_key=api_key)
            self.logger.info(f"Initialized Groq client with model: {model}")
        except Exception as e:
            self.logger.error(f"Error initializing Groq client: {str(e)}")
            raise
    
    def generate_review(self, resume_text: str, job_description: str, 
                       similarity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive resume review.
        
        Args:
            resume_text: Extracted resume text
            job_description: Job description text
            similarity_data: Similarity calculation results
            
        Returns:
            Dictionary containing the complete review
        """
        try:
            # Generate different components of the review
            strengths = self._analyze_strengths(resume_text, job_description, similarity_data)
            weaknesses = self._analyze_weaknesses(resume_text, job_description, similarity_data)
            suggestions = self._generate_suggestions(resume_text, job_description, similarity_data)
            overall_assessment = self._generate_overall_assessment(resume_text, job_description, similarity_data)
            
            review = {
                "match_score": similarity_data.get("similarity_score", 0),
                "match_category": similarity_data.get("match_category", "Unknown"),
                "overall_assessment": overall_assessment,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "improvement_suggestions": suggestions,
                "technical_details": {
                    "similarity_method": similarity_data.get("embedding_method", "unknown"),
                    "cosine_similarity": similarity_data.get("cosine_similarity", 0),
                    "resume_length": len(resume_text),
                    "job_description_length": len(job_description)
                }
            }
            
            return review
            
        except Exception as e:
            self.logger.error(f"Error generating review: {str(e)}")
            raise
    
    def _analyze_strengths(self, resume_text: str, job_description: str, 
                          similarity_data: Dict[str, Any]) -> List[str]:
        """Analyze and identify resume strengths."""
        prompt = f"""
        Analyze the following resume against the job description and identify the TOP 5 STRENGTHS.
        
        Job Description:
        {job_description[:2000]}
        
        Resume:
        {resume_text[:2000]}
        
        Match Score: {similarity_data.get('similarity_score', 0):.1f}%
        
        Instructions:
        - Focus on skills, experiences, and qualifications that directly match the job requirements
        - Highlight unique value propositions
        - Consider both technical and soft skills
        - Be specific and actionable
        - Return ONLY a JSON list of strings, no other text
        
        Format: ["strength 1", "strength 2", "strength 3", "strength 4", "strength 5"]
        """
        
        return self._call_groq_api(prompt, "strengths analysis")
    
    def _analyze_weaknesses(self, resume_text: str, job_description: str, 
                           similarity_data: Dict[str, Any]) -> List[str]:
        """Analyze and identify resume weaknesses."""
        prompt = f"""
        Analyze the following resume against the job description and identify the TOP 5 AREAS FOR IMPROVEMENT.
        
        Job Description:
        {job_description[:2000]}
        
        Resume:
        {resume_text[:2000]}
        
        Match Score: {similarity_data.get('similarity_score', 0):.1f}%
        
        Instructions:
        - Identify missing skills or experiences mentioned in the job description
        - Point out formatting, structure, or presentation issues
        - Note gaps in qualifications or certifications
        - Consider missing quantifiable achievements
        - Be constructive and specific
        - Return ONLY a JSON list of strings, no other text
        
        Format: ["weakness 1", "weakness 2", "weakness 3", "weakness 4", "weakness 5"]
        """
        
        return self._call_groq_api(prompt, "weaknesses analysis")
    
    def _generate_suggestions(self, resume_text: str, job_description: str, 
                            similarity_data: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions."""
        prompt = f"""
        Based on the resume and job description analysis, provide 5 SPECIFIC IMPROVEMENT SUGGESTIONS.
        
        Job Description:
        {job_description[:2000]}
        
        Resume:
        {resume_text[:2000]}
        
        Match Score: {similarity_data.get('similarity_score', 0):.1f}%
        
        Instructions:
        - Provide actionable, specific suggestions for improvement
        - Focus on how to better align with the job requirements
        - Include suggestions for additional skills, certifications, or experiences
        - Consider resume formatting and presentation improvements
        - Suggest quantifiable metrics to add where missing
        - Return ONLY a JSON list of strings, no other text
        
        Format: ["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4", "suggestion 5"]
        """
        
        return self._call_groq_api(prompt, "suggestions generation")
    
    def _generate_overall_assessment(self, resume_text: str, job_description: str, 
                                   similarity_data: Dict[str, Any]) -> str:
        """Generate an overall assessment of the resume-job match."""
        prompt = f"""
        Provide a comprehensive OVERALL ASSESSMENT of how well this resume matches the job description.
        
        Job Description:
        {job_description[:2000]}
        
        Resume:
        {resume_text[:2000]}
        
        Match Score: {similarity_data.get('similarity_score', 0):.1f}%
        Match Category: {similarity_data.get('match_category', 'Unknown')}
        
        Instructions:
        - Write a 3-4 sentence professional assessment
        - Summarize the overall fit between candidate and role
        - Mention the most critical strengths and gaps
        - Provide a realistic hiring recommendation perspective
        - Be balanced and constructive
        - Return ONLY the assessment text, no JSON formatting
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=300,
                temperature=0.3
            )
            
            assessment = response.choices[0].message.content.strip()
            self.logger.debug("Generated overall assessment successfully")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error generating overall assessment: {str(e)}")
            return f"Unable to generate assessment due to API error. Match score: {similarity_data.get('similarity_score', 0):.1f}%"
    
    def _call_groq_api(self, prompt: str, analysis_type: str) -> List[str]:
        """Make a call to Groq API and parse the response."""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                parsed_response = json.loads(content)
                if isinstance(parsed_response, list):
                    self.logger.debug(f"Successfully generated {analysis_type}")
                    return parsed_response[:5]  # Ensure max 5 items
                else:
                    raise ValueError("Response is not a list")
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract from text
                self.logger.warning(f"JSON parsing failed for {analysis_type}, attempting text extraction")
                return self._extract_from_text(content)
                
        except Exception as e:
            self.logger.error(f"Error in Groq API call for {analysis_type}: {str(e)}")
            return [f"Unable to generate {analysis_type} due to API error"]
    
    def _extract_from_text(self, text: str) -> List[str]:
        """Extract list items from plain text response."""
        lines = text.strip().split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            # Remove numbers, bullets, and common prefixes
            line = line.lstrip('0123456789.-• ')
            if line and not line.startswith('[') and not line.startswith('{'):
                items.append(line)
                
        return items[:5]  # Return max 5 items
    
    def generate_quick_score(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Generate a quick compatibility score without full review.
        
        Args:
            resume_text: Resume text
            job_description: Job description text
            
        Returns:
            Dictionary with quick score and basic assessment
        """
        prompt = f"""
        Quickly assess the compatibility between this resume and job description.
        Provide a compatibility score from 0-100 and a brief 2-sentence explanation.
        
        Job Description:
        {job_description[:1500]}
        
        Resume:
        {resume_text[:1500]}
        
        Respond in this exact JSON format:
        {{
            "compatibility_score": <number 0-100>,
            "brief_assessment": "<2 sentence explanation>"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=200,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            
            return {
                "compatibility_score": result.get("compatibility_score", 0),
                "brief_assessment": result.get("brief_assessment", "Unable to assess compatibility"),
                "method": "ai_quick_score"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating quick score: {str(e)}")
            return {
                "compatibility_score": 0,
                "brief_assessment": "Unable to generate quick score due to API error",
                "method": "ai_quick_score",
                "error": str(e)
            }