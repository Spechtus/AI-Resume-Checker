#!/usr/bin/env python3
"""Basic functionality test for AI Resume Checker."""

import sys
from pathlib import Path

def test_text_extraction():
    """Test text extraction functionality."""
    print("Testing text extraction...")
    
    try:
        from resume_checker.extractors import TextExtractor
        
        extractor = TextExtractor()
        result = extractor.extract_text('examples/sample_resume.txt')
        
        assert len(result['text']) > 0, "No text extracted"
        assert result['file_type'] == 'txt', "Wrong file type detected"
        assert 'John Smith' in result['text'], "Expected content not found"
        
        print("✅ Text extraction test passed")
        return True
        
    except Exception as e:
        print(f"❌ Text extraction test failed: {str(e)}")
        return False

def test_similarity_calculation():
    """Test similarity calculation functionality."""
    print("Testing similarity calculation...")
    
    try:
        from resume_checker.similarity import SimilarityCalculator
        from resume_checker.extractors import TextExtractor
        
        extractor = TextExtractor()
        calculator = SimilarityCalculator()
        
        # Load sample files
        resume_data = extractor.extract_text('examples/sample_resume.txt')
        job_data = extractor.extract_text('examples/sample_job_description.txt')
        
        # Calculate similarity
        similarity = calculator.calculate_similarity(resume_data['text'], job_data['text'])
        
        assert 'similarity_score' in similarity, "No similarity score returned"
        assert 0 <= similarity['similarity_score'] <= 100, "Invalid similarity score range"
        assert 'match_category' in similarity, "No match category returned"
        
        print(f"✅ Similarity calculation test passed (Score: {similarity['similarity_score']:.1f}%)")
        return True
        
    except Exception as e:
        print(f"❌ Similarity calculation test failed: {str(e)}")
        return False

def test_text_preprocessing():
    """Test text preprocessing functionality."""
    print("Testing text preprocessing...")
    
    try:
        from resume_checker.utils import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        
        # Test text cleaning
        dirty_text = "This is a TEST with CAPS and   extra   spaces!!!"
        clean_text = preprocessor.clean_text(dirty_text)
        
        assert clean_text == "this is a test with caps and extra spaces!!!", "Text cleaning failed"
        
        # Test keyword extraction
        keywords = preprocessor.extract_keywords("Python programming with Django framework")
        
        assert 'python' in keywords, "Expected keyword not found"
        assert 'django' in keywords, "Expected keyword not found"
        assert len(keywords) > 0, "No keywords extracted"
        
        print("✅ Text preprocessing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Text preprocessing test failed: {str(e)}")
        return False

def test_main_integration():
    """Test main ResumeChecker integration (without AI review)."""
    print("Testing main integration...")
    
    try:
        from resume_checker import ResumeChecker
        
        # Initialize without API key (will skip AI review)
        checker = ResumeChecker()
        
        # Test quick score
        result = checker.quick_score(
            'examples/sample_resume.txt',
            'examples/sample_job_description.txt'
        )
        
        assert 'similarity_score' in result, "No similarity score in quick result"
        assert 'match_category' in result, "No match category in quick result"
        
        print(f"✅ Main integration test passed (Quick Score: {result['similarity_score']:.1f}%)")
        return True
        
    except Exception as e:
        print(f"❌ Main integration test failed: {str(e)}")
        return False

def test_configuration():
    """Test configuration functionality."""
    print("Testing configuration...")
    
    try:
        from resume_checker.utils import Config
        
        config = Config()
        
        # Test getting and setting values
        config.set('test_key', 'test_value')
        assert config.get('test_key') == 'test_value', "Config set/get failed"
        
        # Test configuration validation
        validation = config.validate_api_keys()
        assert isinstance(validation, dict), "Validation should return dict"
        assert 'groq_api_key' in validation, "Should validate Groq API key"
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False

def main():
    """Run all basic functionality tests."""
    print("🚀 Running AI Resume Checker Basic Functionality Tests\n")
    
    # Check if sample files exist
    if not Path('examples/sample_resume.txt').exists():
        print("❌ Sample resume file not found")
        return False
    
    if not Path('examples/sample_job_description.txt').exists():
        print("❌ Sample job description file not found")
        return False
    
    # Run tests
    tests = [
        test_configuration,
        test_text_extraction,
        test_text_preprocessing,
        test_similarity_calculation,
        test_main_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}")
            failed += 1
        print()  # Empty line for readability
    
    # Summary
    print("=" * 50)
    print(f"Tests completed: {passed + failed}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All basic functionality tests passed!")
        print("The AI Resume Checker core system is working correctly.")
        print("\nNote: AI review functionality requires a Groq API key.")
        print("Set GROQ_API_KEY environment variable to test AI features.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)