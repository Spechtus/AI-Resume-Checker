"""Text preprocessing utilities for resume and job description analysis."""

import re
import logging
from typing import List, Dict, Any, Optional


class TextPreprocessor:
    """Handles text preprocessing for resume and job description analysis."""
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.logger = logging.getLogger(__name__)
        
        # Common patterns to clean
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.extra_whitespace_pattern = re.compile(r'\s+')
        
    def clean_text(self, text: str, remove_personal_info: bool = False) -> str:
        """
        Clean and normalize text for analysis.
        
        Args:
            text: Raw text to clean
            remove_personal_info: Whether to remove email, phone, etc.
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove personal information if requested
        if remove_personal_info:
            text = self.email_pattern.sub('[EMAIL]', text)
            text = self.phone_pattern.sub('[PHONE]', text)
            text = self.url_pattern.sub('[URL]', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', ' ', text)
        
        # Replace multiple whitespace with single space
        text = self.extra_whitespace_pattern.sub(' ', text)
        
        # Remove extra newlines and strip
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract common resume sections from text.
        
        Args:
            text: Resume text to analyze
            
        Returns:
            Dictionary with extracted sections
        """
        sections = {
            'summary': '',
            'experience': '',
            'education': '',
            'skills': '',
            'projects': '',
            'other': ''
        }
        
        # Common section headers (case-insensitive)
        patterns = {
            'summary': r'(summary|profile|objective|about)[\s\n]*:?',
            'experience': r'(experience|work\s+history|employment|career)[\s\n]*:?',
            'education': r'(education|academic|qualification|degree)[\s\n]*:?',
            'skills': r'(skills|competencies|technical\s+skills|expertise)[\s\n]*:?',
            'projects': r'(projects|portfolio|work\s+samples)[\s\n]*:?'
        }
        
        text_lower = text.lower()
        lines = text.split('\n')
        current_section = 'other'
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            section_found = False
            for section_name, pattern in patterns.items():
                if re.search(pattern, line_lower):
                    current_section = section_name
                    section_found = True
                    break
            
            # Add content to current section (skip header lines)
            if not section_found and line.strip():
                if sections[current_section]:
                    sections[current_section] += ' '
                sections[current_section] += line.strip()
        
        # Clean sections
        for section in sections:
            sections[section] = self.clean_text(sections[section])
        
        return sections
    
    def extract_keywords(self, text: str, min_length: int = 2) -> List[str]:
        """
        Extract relevant keywords from text.
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum word length to consider
            
        Returns:
            List of keywords
        """
        # Clean text first
        text = self.clean_text(text)
        
        # Common stop words to exclude
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'myself', 'yourself',
            'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter and collect keywords
        keywords = []
        for word in words:
            word = word.lower()
            if (len(word) >= min_length and 
                word not in stop_words and 
                not word.isdigit()):
                keywords.append(word)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords
    
    def normalize_text_for_similarity(self, text: str) -> str:
        """
        Normalize text specifically for similarity comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Clean text and remove personal info
        text = self.clean_text(text, remove_personal_info=True)
        
        # Additional normalization for similarity
        # Remove common resume/job posting boilerplate
        boilerplate_patterns = [
            r'\b(resume|cv|curriculum vitae)\b',
            r'\b(equal opportunity employer)\b',
            r'\b(eoe)\b',
            r'\b(apply now|apply today)\b',
            r'\b(contact information|contact details)\b'
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces again
        text = self.extra_whitespace_pattern.sub(' ', text).strip()
        
        return text
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get basic statistics about the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'average_sentence_length': len(words) / len(sentences) if sentences else 0
        }