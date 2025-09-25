#!/usr/bin/env python3
"""Command-line interface for the AI Resume Checker."""

import click
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from resume_checker import ResumeChecker
from resume_checker.utils import Config

# Initialize console for rich output
console = Console() if RICH_AVAILABLE else None


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """AI Resume Checker - Analyze resumes against job descriptions using AI."""
    pass


@cli.command()
@click.argument('resume_path', type=click.Path(exists=True))
@click.argument('job_description')
@click.option('--output', '-o', help='Output file path (JSON or TXT)')
@click.option('--no-ai-review', is_flag=True, help='Skip AI-powered review generation')
@click.option('--groq-api-key', help='Groq API key (overrides environment variable)')
@click.option('--similarity-model', help='Similarity model to use')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(resume_path: str, job_description: str, output: Optional[str],
           no_ai_review: bool, groq_api_key: Optional[str], 
           similarity_model: Optional[str], verbose: bool):
    """
    Analyze a resume against a job description.
    
    RESUME_PATH: Path to the resume file (PDF, DOCX, or TXT)
    JOB_DESCRIPTION: Job description text or path to job description file
    """
    try:
        # Setup configuration
        config = Config()
        if similarity_model:
            config.set('similarity_model', similarity_model)
        
        if verbose:
            config.set('log_level', 'DEBUG')
        
        # Initialize checker
        checker = ResumeChecker(config=config, groq_api_key=groq_api_key)
        
        # Display progress
        _print_info("Starting resume analysis...")
        _print_info(f"Resume: {resume_path}")
        
        if Path(job_description).exists():
            _print_info(f"Job Description: {job_description} (file)")
        else:
            _print_info(f"Job Description: {len(job_description)} characters (text)")
        
        # Perform analysis
        results = checker.analyze_resume(
            resume_path=resume_path,
            job_description=job_description,
            include_ai_review=not no_ai_review
        )
        
        # Display results
        _display_results(results)
        
        # Save results if output specified
        if output:
            checker.save_results(results, output)
            _print_success(f"Results saved to: {output}")
        
    except Exception as e:
        _print_error(f"Error: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('resume_path', type=click.Path(exists=True))
@click.argument('job_description')
@click.option('--groq-api-key', help='Groq API key (overrides environment variable)')
def quick_score(resume_path: str, job_description: str, groq_api_key: Optional[str]):
    """
    Get a quick compatibility score without full analysis.
    
    RESUME_PATH: Path to the resume file
    JOB_DESCRIPTION: Job description text or path to job description file
    """
    try:
        config = Config()
        checker = ResumeChecker(config=config, groq_api_key=groq_api_key)
        
        _print_info("Calculating quick score...")
        
        results = checker.quick_score(resume_path, job_description)
        
        # Display quick results
        _print_success(f"Similarity Score: {results['similarity_score']:.1f}%")
        _print_info(f"Match Category: {results['match_category']}")
        
        if 'ai_score' in results and results['ai_score']:
            ai_score = results['ai_score']
            if 'compatibility_score' in ai_score:
                _print_info(f"AI Compatibility Score: {ai_score['compatibility_score']}%")
                _print_info(f"AI Assessment: {ai_score.get('brief_assessment', 'N/A')}")
        
    except Exception as e:
        _print_error(f"Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('resume_paths', nargs=-1, required=True)
@click.argument('job_description')
@click.option('--output', '-o', help='Output file path for results')
@click.option('--include-ai-review', is_flag=True, help='Include AI reviews (slower)')
@click.option('--groq-api-key', help='Groq API key')
def batch(resume_paths, job_description: str, output: Optional[str],
          include_ai_review: bool, groq_api_key: Optional[str]):
    """
    Analyze multiple resumes against a job description.
    
    RESUME_PATHS: Paths to resume files
    JOB_DESCRIPTION: Job description text or path to job description file
    """
    try:
        config = Config()
        checker = ResumeChecker(config=config, groq_api_key=groq_api_key)
        
        _print_info(f"Analyzing {len(resume_paths)} resumes...")
        
        # Create job descriptions list (same for all)
        job_descriptions = [job_description] * len(resume_paths)
        
        results = checker.batch_analyze(
            resume_paths=list(resume_paths),
            job_descriptions=job_descriptions,
            include_ai_review=include_ai_review
        )
        
        # Display batch results
        _display_batch_results(results)
        
        # Save results if output specified
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            _print_success(f"Batch results saved to: {output}")
        
    except Exception as e:
        _print_error(f"Error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def extract(file_path: str):
    """Extract text from a resume file."""
    try:
        from resume_checker.extractors import TextExtractor
        
        extractor = TextExtractor()
        result = extractor.extract_text(file_path)
        
        _print_info(f"File: {file_path}")
        _print_info(f"Type: {result['file_type']}")
        _print_info(f"Extraction method: {result['extraction_method']}")
        _print_info(f"Text length: {len(result['text'])} characters")
        
        print("\n--- EXTRACTED TEXT ---")
        print(result['text'])
        
    except Exception as e:
        _print_error(f"Error: {str(e)}")
        sys.exit(1)


@cli.command()
def check_setup():
    """Check if the system is properly configured."""
    _print_info("Checking AI Resume Checker setup...")
    
    issues = []
    
    # Check API keys
    config = Config()
    api_validation = config.validate_api_keys()
    
    if not api_validation['groq_api_key']:
        issues.append("❌ Groq API key not found (set GROQ_API_KEY environment variable)")
    else:
        _print_success("✅ Groq API key found")
    
    # Check dependencies
    try:
        import PyPDF2
        _print_success("✅ PyPDF2 available")
    except ImportError:
        issues.append("❌ PyPDF2 not installed (pip install PyPDF2)")
    
    try:
        import pdfplumber
        _print_success("✅ pdfplumber available")
    except ImportError:
        issues.append("❌ pdfplumber not installed (pip install pdfplumber)")
    
    try:
        from docx import Document
        _print_success("✅ python-docx available")
    except ImportError:
        issues.append("❌ python-docx not installed (pip install python-docx)")
    
    try:
        from groq import Groq
        _print_success("✅ Groq client available")
    except ImportError:
        issues.append("❌ Groq client not installed (pip install groq)")
    
    try:
        from sentence_transformers import SentenceTransformer
        _print_success("✅ sentence-transformers available")
    except ImportError:
        issues.append("⚠️  sentence-transformers not installed (will use fallback method)")
    
    if issues:
        _print_error("\nSetup Issues Found:")
        for issue in issues:
            print(f"  {issue}")
        print("\nInstall missing dependencies with: pip install -r requirements.txt")
    else:
        _print_success("\n🎉 All systems ready! You can start analyzing resumes.")


def _print_info(message: str):
    """Print info message."""
    if RICH_AVAILABLE:
        console.print(f"ℹ️  {message}", style="blue")
    else:
        print(f"INFO: {message}")


def _print_success(message: str):
    """Print success message."""
    if RICH_AVAILABLE:
        console.print(f"✅ {message}", style="green")
    else:
        print(f"SUCCESS: {message}")


def _print_error(message: str):
    """Print error message."""
    if RICH_AVAILABLE:
        console.print(f"❌ {message}", style="red")
    else:
        print(f"ERROR: {message}")


def _display_results(results: dict):
    """Display analysis results in a formatted way."""
    if not RICH_AVAILABLE:
        _display_results_plain(results)
        return
    
    # Match score
    similarity = results.get('similarity_analysis', {})
    score = similarity.get('similarity_score', 0)
    category = similarity.get('match_category', 'Unknown')
    
    # Create score panel
    score_text = Text(f"{score:.1f}%", style="bold")
    if score >= 70:
        score_text.style = "bold green"
    elif score >= 50:
        score_text.style = "bold yellow"
    else:
        score_text.style = "bold red"
    
    console.print(Panel(
        f"Match Score: {score_text}\nCategory: {category}",
        title="📊 Resume Match Analysis",
        border_style="blue"
    ))
    
    # AI Review
    ai_review = results.get('ai_review')
    if ai_review and 'error' not in ai_review:
        
        # Overall assessment
        if 'overall_assessment' in ai_review:
            console.print(Panel(
                ai_review['overall_assessment'],
                title="🎯 Overall Assessment",
                border_style="green"
            ))
        
        # Strengths
        if 'strengths' in ai_review and ai_review['strengths']:
            strengths_text = "\n".join(f"• {s}" for s in ai_review['strengths'])
            console.print(Panel(
                strengths_text,
                title="💪 Strengths",
                border_style="green"
            ))
        
        # Weaknesses
        if 'weaknesses' in ai_review and ai_review['weaknesses']:
            weaknesses_text = "\n".join(f"• {w}" for w in ai_review['weaknesses'])
            console.print(Panel(
                weaknesses_text,
                title="⚠️  Areas for Improvement",
                border_style="yellow"
            ))
        
        # Suggestions
        if 'improvement_suggestions' in ai_review and ai_review['improvement_suggestions']:
            suggestions_text = "\n".join(f"• {s}" for s in ai_review['improvement_suggestions'])
            console.print(Panel(
                suggestions_text,
                title="💡 Improvement Suggestions",
                border_style="cyan"
            ))


def _display_results_plain(results: dict):
    """Display results in plain text format."""
    print("\n=== RESUME ANALYSIS RESULTS ===")
    
    similarity = results.get('similarity_analysis', {})
    score = similarity.get('similarity_score', 0)
    category = similarity.get('match_category', 'Unknown')
    
    print(f"Match Score: {score:.1f}%")
    print(f"Match Category: {category}")
    
    ai_review = results.get('ai_review')
    if ai_review and 'error' not in ai_review:
        
        if 'overall_assessment' in ai_review:
            print(f"\nOverall Assessment:\n{ai_review['overall_assessment']}")
        
        if 'strengths' in ai_review and ai_review['strengths']:
            print("\nStrengths:")
            for i, strength in enumerate(ai_review['strengths'], 1):
                print(f"{i}. {strength}")
        
        if 'weaknesses' in ai_review and ai_review['weaknesses']:
            print("\nAreas for Improvement:")
            for i, weakness in enumerate(ai_review['weaknesses'], 1):
                print(f"{i}. {weakness}")
        
        if 'improvement_suggestions' in ai_review and ai_review['improvement_suggestions']:
            print("\nImprovement Suggestions:")
            for i, suggestion in enumerate(ai_review['improvement_suggestions'], 1):
                print(f"{i}. {suggestion}")


def _display_batch_results(results: list):
    """Display batch analysis results."""
    if not RICH_AVAILABLE:
        _display_batch_results_plain(results)
        return
    
    table = Table(title="Batch Analysis Results")
    table.add_column("Resume", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Category", style="green")
    table.add_column("Status")
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            table.add_row(
                f"Resume {i}",
                "ERROR",
                "ERROR",
                f"❌ {result['error'][:50]}..."
            )
        else:
            similarity = result.get('similarity_analysis', {})
            score = similarity.get('similarity_score', 0)
            category = similarity.get('match_category', 'Unknown')
            
            score_style = "green" if score >= 70 else "yellow" if score >= 50 else "red"
            
            table.add_row(
                f"Resume {i}",
                f"[{score_style}]{score:.1f}%[/{score_style}]",
                category,
                "✅ Complete"
            )
    
    console.print(table)


def _display_batch_results_plain(results: list):
    """Display batch results in plain text."""
    print("\n=== BATCH ANALYSIS RESULTS ===")
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            print(f"Resume {i}: ERROR - {result['error']}")
        else:
            similarity = result.get('similarity_analysis', {})
            score = similarity.get('similarity_score', 0)
            category = similarity.get('match_category', 'Unknown')
            print(f"Resume {i}: {score:.1f}% ({category})")


if __name__ == '__main__':
    cli()