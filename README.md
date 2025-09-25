# AI Resume Checker 🚀

An AI-powered system that analyzes resumes against job descriptions, computes Resume-Job Match Scores, and generates detailed strengths, weaknesses, and improvement suggestions using Groq and Hugging Face APIs.

## Features ✨

### Part 1: Text Extraction
- **Multi-format Support**: Extract text from PDF, DOCX, and TXT files
- **Robust Processing**: Advanced text extraction with fallback methods
- **Metadata Collection**: File statistics and extraction details

### Part 2: Resume-Job Similarity Calculation
- **Advanced Embeddings**: Uses Sentence Transformers for semantic similarity
- **Fallback Methods**: TF-IDF vectorization when transformers unavailable
- **Comprehensive Metrics**: Cosine similarity, match categories, and detailed analysis

### Part 3: AI-Powered Resume Review
- **Groq AI Integration**: Leverages Groq's powerful language models
- **Detailed Analysis**: Identifies strengths, weaknesses, and improvement areas
- **Actionable Suggestions**: Specific recommendations for resume enhancement

## Quick Start 🚀

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Spechtus/AI-Resume-Checker.git
cd AI-Resume-Checker
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

4. **Get your Groq API key:**
   - Sign up at [Groq](https://console.groq.com/)
   - Get your API key from the dashboard
   - Add it to your `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Basic Usage

#### Command Line Interface

**Analyze a resume:**
```bash
python cli.py analyze examples/sample_resume.txt examples/sample_job_description.txt
```

**Quick compatibility score:**
```bash
python cli.py quick-score examples/sample_resume.txt examples/sample_job_description.txt
```

**Extract text from a file:**
```bash
python cli.py extract examples/sample_resume.txt
```

**Check system setup:**
```bash
python cli.py check-setup
```

#### Python API

```python
from resume_checker import ResumeChecker

# Initialize the checker
checker = ResumeChecker(groq_api_key="your_groq_api_key")

# Analyze a resume
results = checker.analyze_resume(
    resume_path="path/to/resume.pdf",
    job_description="Job description text or path to file"
)

# Print match score
print(f"Match Score: {results['similarity_analysis']['similarity_score']:.1f}%")

# Get AI review
ai_review = results['ai_review']
print("Strengths:", ai_review['strengths'])
print("Improvements:", ai_review['improvement_suggestions'])
```

## Detailed Usage 📖

### Command Line Options

#### `analyze` command
```bash
python cli.py analyze [OPTIONS] RESUME_PATH JOB_DESCRIPTION

Options:
  --output, -o PATH          Save results to file (JSON or TXT)
  --no-ai-review            Skip AI-powered review generation
  --groq-api-key TEXT       Groq API key (overrides env var)
  --similarity-model TEXT   Similarity model to use
  --verbose, -v             Verbose output
```

#### `batch` command
```bash
python cli.py batch [OPTIONS] RESUME_PATHS... JOB_DESCRIPTION

Options:
  --output, -o PATH         Output file for batch results
  --include-ai-review       Include AI reviews (slower)
  --groq-api-key TEXT       Groq API key
```

### Python API Examples

#### Basic Analysis
```python
from resume_checker import ResumeChecker
from resume_checker.utils import Config

# Configure the system
config = Config()
config.set('similarity_model', 'sentence-transformers/all-MiniLM-L6-v2')
config.set('groq_model', 'llama3-8b-8192')

# Initialize checker
checker = ResumeChecker(config=config)

# Analyze resume
results = checker.analyze_resume(
    resume_path="resume.pdf",
    job_description="job_description.txt",
    include_ai_review=True
)

# Access results
similarity_score = results['similarity_analysis']['similarity_score']
match_category = results['similarity_analysis']['match_category']
ai_strengths = results['ai_review']['strengths']
ai_suggestions = results['ai_review']['improvement_suggestions']
```

#### Batch Processing
```python
# Analyze multiple resumes
resume_paths = ["resume1.pdf", "resume2.docx", "resume3.txt"]
job_descriptions = ["job1.txt", "job2.txt", "job3.txt"]

batch_results = checker.batch_analyze(
    resume_paths=resume_paths,
    job_descriptions=job_descriptions,
    include_ai_review=True
)

# Process results
for i, result in enumerate(batch_results):
    if 'error' not in result:
        score = result['similarity_analysis']['similarity_score']
        print(f"Resume {i+1}: {score:.1f}% match")
```

#### Text Extraction Only
```python
from resume_checker.extractors import TextExtractor

extractor = TextExtractor()
result = extractor.extract_text("resume.pdf")

print(f"Extracted {len(result['text'])} characters")
print(f"File type: {result['file_type']}")
print(f"Method: {result['extraction_method']}")
```

#### Similarity Calculation Only
```python
from resume_checker.similarity import SimilarityCalculator

calculator = SimilarityCalculator()
similarity = calculator.calculate_similarity(resume_text, job_text)

print(f"Cosine similarity: {similarity['cosine_similarity']:.3f}")
print(f"Match score: {similarity['similarity_score']:.1f}%")
print(f"Category: {similarity['match_category']}")
```

## Configuration ⚙️

### Environment Variables

The system can be configured using environment variables or a `.env` file:

```bash
# API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192

# Similarity Model
SIMILARITY_MODEL=sentence-transformers/all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.5

# Processing Settings
MAX_TEXT_LENGTH=10000
REMOVE_PERSONAL_INFO=true

# Review Settings
MAX_REVIEW_LENGTH=2000
INCLUDE_SUGGESTIONS=true
SUGGESTION_COUNT=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=resume_checker.log
```

### Programmatic Configuration

```python
from resume_checker.utils import Config

config = Config()

# Set API keys
config.set('groq_api_key', 'your_key_here')

# Configure models
config.set('similarity_model', 'sentence-transformers/all-mpnet-base-v2')
config.set('groq_model', 'llama3-70b-8192')

# Configure processing
config.set('max_text_length', 15000)
config.set('remove_personal_info', False)

# Use with ResumeChecker
checker = ResumeChecker(config=config)
```

## Architecture 🏗️

```
AI-Resume-Checker/
├── resume_checker/           # Main package
│   ├── extractors/          # Text extraction (Part 1)
│   │   └── text_extractor.py
│   ├── similarity/          # Similarity calculation (Part 2)
│   │   └── similarity_calculator.py
│   ├── reviewers/           # AI review generation (Part 3)
│   │   └── ai_reviewer.py
│   ├── utils/               # Utilities
│   │   ├── config.py
│   │   └── text_preprocessor.py
│   └── main.py              # Main orchestrator
├── cli.py                   # Command-line interface
├── examples/                # Sample files
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

### Key Components

1. **TextExtractor**: Handles PDF, DOCX, and TXT file processing
2. **SimilarityCalculator**: Computes semantic similarity using various methods
3. **AIReviewer**: Generates detailed reviews using Groq API
4. **Config**: Manages configuration and API keys
5. **TextPreprocessor**: Cleans and normalizes text for analysis
6. **ResumeChecker**: Main orchestrator class

## Dependencies 📦

### Core Dependencies
- **PyPDF2** & **pdfplumber**: PDF text extraction
- **python-docx**: DOCX file processing
- **groq**: Groq API client for AI reviews
- **transformers**: Hugging Face transformers
- **sentence-transformers**: Advanced text embeddings
- **scikit-learn**: Machine learning utilities
- **click**: Command-line interface
- **rich**: Enhanced terminal output

### Optional Dependencies
- **torch**: PyTorch for advanced models
- **python-dotenv**: Environment variable management

## Examples and Use Cases 🎯

### 1. HR Screening
```python
# Screen multiple candidates
candidates = ["resume1.pdf", "resume2.pdf", "resume3.pdf"]
job_desc = "Senior Python Developer position..."

results = checker.batch_analyze(candidates, [job_desc] * len(candidates))

# Rank candidates by match score
ranked = sorted(results, 
                key=lambda x: x['similarity_analysis']['similarity_score'], 
                reverse=True)

for i, result in enumerate(ranked):
    score = result['similarity_analysis']['similarity_score']
    print(f"Rank {i+1}: {score:.1f}% match")
```

### 2. Resume Optimization
```python
# Analyze your resume against a job
results = checker.analyze_resume("my_resume.pdf", job_description)

# Get improvement suggestions
suggestions = results['ai_review']['improvement_suggestions']
print("Ways to improve your resume:")
for suggestion in suggestions:
    print(f"• {suggestion}")
```

### 3. Job Market Analysis
```python
# Test resume against multiple jobs
jobs = ["job1.txt", "job2.txt", "job3.txt"]
my_resume = "my_resume.pdf"

scores = []
for job in jobs:
    result = checker.quick_score(my_resume, job)
    scores.append(result['similarity_score'])

best_match = max(enumerate(scores), key=lambda x: x[1])
print(f"Best job match: Job {best_match[0]+1} ({best_match[1]:.1f}%)")
```

## Troubleshooting 🔧

### Common Issues

1. **"Groq API key not found"**
   - Set the `GROQ_API_KEY` environment variable
   - Or pass it directly: `ResumeChecker(groq_api_key="your_key")`

2. **PDF extraction fails**
   - Ensure PyPDF2 and pdfplumber are installed
   - Some PDFs may be image-based and require OCR

3. **Similarity calculation slow**
   - Use a smaller model: `SIMILARITY_MODEL=sentence-transformers/all-MiniLM-L6-v2`
   - Or fallback to TF-IDF by not installing sentence-transformers

4. **AI review generation fails**
   - Check your Groq API key and quota
   - Ensure text lengths are within limits

### Performance Tips

- Use smaller similarity models for faster processing
- Enable text preprocessing to reduce input size
- Use quick_score for initial screening
- Batch process multiple resumes for efficiency

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- [Groq](https://groq.com/) for providing fast AI inference
- [Hugging Face](https://huggingface.co/) for transformer models
- [Sentence Transformers](https://www.sbert.net/) for semantic embeddings

---

**Happy Resume Checking! 🎉**