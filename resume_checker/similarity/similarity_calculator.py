"""Similarity calculation module for resume-job matching."""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Try to import sentence transformers for advanced embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import transformers for fallback embeddings
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SimilarityCalculator:
    """Calculates similarity between resume and job description texts."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 use_tfidf_fallback: bool = True):
        """
        Initialize the similarity calculator.
        
        Args:
            model_name: Name of the model to use for embeddings
            use_tfidf_fallback: Whether to use TF-IDF as fallback if transformers unavailable
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.use_tfidf_fallback = use_tfidf_fallback
        
        # Initialize the model
        self.model = None
        self.tokenizer = None
        self.embedding_method = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.model = SentenceTransformer(self.model_name)
                self.embedding_method = "sentence_transformers"
                self.logger.info(f"Initialized SentenceTransformer model: {self.model_name}")
                
            elif TRANSFORMERS_AVAILABLE:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.embedding_method = "transformers"
                self.logger.info(f"Initialized Transformers model: {self.model_name}")
                
            elif self.use_tfidf_fallback:
                self.model = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.embedding_method = "tfidf"
                self.logger.warning("Using TF-IDF fallback method for similarity calculation")
                
            else:
                raise ImportError("No suitable embedding method available")
                
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            if self.use_tfidf_fallback and self.embedding_method != "tfidf":
                self.logger.info("Falling back to TF-IDF method")
                self.model = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.embedding_method = "tfidf"
            else:
                raise
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        try:
            if self.embedding_method == "sentence_transformers":
                return self.model.encode(texts)
                
            elif self.embedding_method == "transformers":
                return self._get_transformers_embeddings(texts)
                
            elif self.embedding_method == "tfidf":
                return self._get_tfidf_embeddings(texts)
                
            else:
                raise ValueError(f"Unknown embedding method: {self.embedding_method}")
                
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {str(e)}")
            raise
    
    def _get_transformers_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using transformers library."""
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _get_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using TF-IDF."""
        if not hasattr(self.model, 'vocabulary_'):
            # First time - fit the vectorizer
            embeddings = self.model.fit_transform(texts)
        else:
            # Already fitted - just transform
            embeddings = self.model.transform(texts)
        
        return embeddings.toarray()
    
    def calculate_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text (e.g., resume)
            text2: Second text (e.g., job description)
            
        Returns:
            Dictionary with similarity metrics
        """
        try:
            # Get embeddings
            embeddings = self.get_embeddings([text1, text2])
            
            if len(embeddings) != 2:
                raise ValueError("Failed to generate embeddings for both texts")
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
            cosine_sim = float(similarity_matrix[0][0])
            
            # Additional metrics
            result = {
                "cosine_similarity": cosine_sim,
                "similarity_score": self._normalize_score(cosine_sim),
                "match_category": self._categorize_match(cosine_sim),
                "embedding_method": self.embedding_method,
                "text1_length": len(text1),
                "text2_length": len(text2)
            }
            
            # Add detailed analysis if using advanced methods
            if self.embedding_method in ["sentence_transformers", "transformers"]:
                result.update(self._detailed_analysis(text1, text2, embeddings))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            raise
    
    def _normalize_score(self, cosine_sim: float) -> float:
        """
        Normalize cosine similarity to a 0-100 scale.
        
        Args:
            cosine_sim: Cosine similarity value (-1 to 1)
            
        Returns:
            Normalized score (0 to 100)
        """
        # Convert from [-1, 1] to [0, 100]
        return ((cosine_sim + 1) / 2) * 100
    
    def _categorize_match(self, cosine_sim: float) -> str:
        """
        Categorize the match quality based on similarity score.
        
        Args:
            cosine_sim: Cosine similarity value
            
        Returns:
            Match category string
        """
        score = self._normalize_score(cosine_sim)
        
        if score >= 80:
            return "Excellent Match"
        elif score >= 65:
            return "Good Match"
        elif score >= 50:
            return "Fair Match"
        elif score >= 35:
            return "Poor Match"
        else:
            return "Very Poor Match"
    
    def _detailed_analysis(self, text1: str, text2: str, 
                          embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Perform detailed analysis of the similarity.
        
        Args:
            text1: First text
            text2: Second text
            embeddings: Pre-computed embeddings
            
        Returns:
            Dictionary with detailed metrics
        """
        detailed = {}
        
        try:
            # Calculate additional similarity metrics
            # Euclidean distance
            euclidean_dist = np.linalg.norm(embeddings[0] - embeddings[1])
            detailed["euclidean_distance"] = float(euclidean_dist)
            
            # Manhattan distance
            manhattan_dist = np.sum(np.abs(embeddings[0] - embeddings[1]))
            detailed["manhattan_distance"] = float(manhattan_dist)
            
            # Embedding statistics
            detailed["embedding_dimensions"] = embeddings.shape[1]
            detailed["text1_embedding_norm"] = float(np.linalg.norm(embeddings[0]))
            detailed["text2_embedding_norm"] = float(np.linalg.norm(embeddings[1]))
            
        except Exception as e:
            self.logger.warning(f"Error in detailed analysis: {str(e)}")
        
        return detailed
    
    def batch_similarity(self, resume_texts: List[str], 
                        job_descriptions: List[str]) -> List[Dict[str, Any]]:
        """
        Calculate similarity for multiple resume-job pairs.
        
        Args:
            resume_texts: List of resume texts
            job_descriptions: List of job description texts
            
        Returns:
            List of similarity results
        """
        if len(resume_texts) != len(job_descriptions):
            raise ValueError("Number of resumes and job descriptions must match")
        
        results = []
        for resume, job_desc in zip(resume_texts, job_descriptions):
            try:
                similarity = self.calculate_similarity(resume, job_desc)
                results.append(similarity)
            except Exception as e:
                self.logger.error(f"Error in batch similarity calculation: {str(e)}")
                results.append({
                    "error": str(e),
                    "cosine_similarity": 0.0,
                    "similarity_score": 0.0,
                    "match_category": "Error"
                })
        
        return results
    
    def get_top_matches(self, resume_text: str, job_descriptions: List[str], 
                       top_k: int = 5) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Find top matching job descriptions for a resume.
        
        Args:
            resume_text: Resume text
            job_descriptions: List of job description texts
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_result) tuples, sorted by similarity
        """
        similarities = []
        
        for i, job_desc in enumerate(job_descriptions):
            try:
                similarity = self.calculate_similarity(resume_text, job_desc)
                similarities.append((i, similarity))
            except Exception as e:
                self.logger.error(f"Error calculating similarity for job {i}: {str(e)}")
                similarities.append((i, {
                    "error": str(e),
                    "cosine_similarity": 0.0,
                    "similarity_score": 0.0
                }))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1].get("similarity_score", 0), reverse=True)
        
        return similarities[:top_k]