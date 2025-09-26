# 📄 Resume-to-Job Matching System with ATS Mode

An end-to-end NLP-powered system that evaluates resume-job compatibility and generates ATS-optimized versions to improve application success rates.

## 🎯 Features

### Core Capabilities
- **Semantic Match Scoring**: Computes similarity between resume and job description using transformer embeddings
- **Gap Analysis**: Identifies missing skills, keywords, and experience requirements
- **ATS Optimization**: Generates keyword-optimized resume versions without keyword stuffing
- **Section-Level Analysis**: Provides detailed insights for each resume section
- **Action Verb Enhancement**: Analyzes and improves use of impactful action verbs
- **Visual Analytics**: Interactive charts and metrics for comprehensive understanding

### Analysis Modes
1. **Quick Match**: Fast compatibility score and basic metrics
2. **Detailed Analysis**: Deep dive into gaps, skills, and improvements
3. **ATS Optimization**: Generate enhanced resume with improved keyword coverage

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended for optimal performance)

### Installation

#### Option 1: Automated Installation (Windows)
```bash
git clone https://github.com/yourusername/resume-ats-matcher.git
cd resume-ats-matcher
install.bat
```

#### Option 2: Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-ats-matcher.git
cd resume-ats-matcher
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

**For Python 3.8-3.12:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**For Python 3.13+ or if you encounter errors:**
```bash
# Install basic dependencies first
pip install -r requirements-simple.txt

# Then install ML packages
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers transformers

# Finally, install spaCy
pip install spacy
python -m spacy download en_core_web_sm
```

#### Option 3: Lite Version (No ML Dependencies)
If you're having trouble with ML packages, use the lightweight version:
```bash
pip install -r requirements-simple.txt
streamlit run app_lite.py
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Documents
- Upload your resume (PDF, DOCX, or TXT format)
- Upload the job description you're targeting

### 2. Select Analysis Mode
- **Quick Match**: Get instant compatibility score
- **Detailed Analysis**: Comprehensive breakdown with visualizations
- **ATS Optimization**: Generate improved resume version

### 3. Review Results
- Overall match percentage
- Section-wise scoring
- Missing keywords identification
- Skills gap analysis
- Action verb density

### 4. Download Optimized Resume
- Get ATS-friendly version with improved keyword coverage
- Download optimization report with metrics

## 🏗️ Architecture

### Component Structure
```
src/
├── text_extractor.py     # Document parsing and section extraction
├── semantic_matcher.py   # SBERT-based similarity computation
├── keyword_analyzer.py   # Keyword extraction and gap analysis
├── ats_optimizer.py      # Resume enhancement engine
├── visualizer.py         # Charts and visual analytics
└── config.py            # Configuration and settings
```

### Key Technologies
- **NLP Models**: Sentence-BERT for semantic similarity
- **Text Processing**: spaCy for NER and linguistic analysis
- **Keyword Extraction**: RAKE algorithm + custom patterns
- **Web Framework**: Streamlit for interactive UI
- **Visualization**: Plotly for interactive charts

## 📊 Metrics Explained

### Match Score
- Weighted average of section-wise similarities
- Sections weighted by importance (experience: 30%, skills: 35%, etc.)

### Keyword Coverage
- Percentage of job keywords found in resume
- Identifies critical missing keywords

### ATS Optimization Score
- Measures improvement in keyword density
- Tracks action verb usage
- Monitors quantifiable achievements

## 🔧 Configuration

Edit `src/config.py` to customize:
- Model selection (default: `all-MiniLM-L6-v2`)
- Section weights for scoring
- Keyword thresholds
- Skill categories

## 📁 Sample Data

Example files provided in `data/`:
- `sample_resume.txt`: Software engineer resume
- `sample_job_description.txt`: Full-stack developer JD

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Model download fails
```bash
# Solution: Manually download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Issue**: spaCy model not found
```bash
# Solution: Re-download language model
python -m spacy download en_core_web_sm --force
```

**Issue**: Memory error with large documents
```
# Solution: Increase swap space or use smaller model in config.py
```

## 📈 Performance

- Resume processing: ~2-3 seconds
- Match calculation: ~1 second
- ATS optimization: ~3-5 seconds
- Supports documents up to 10,000 words

## 🔒 Privacy

- All processing happens locally
- No data sent to external servers
- Uploaded files deleted after processing

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Streamlit team for the excellent framework
- spaCy for NLP capabilities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This tool provides suggestions to improve resume-job matching. Always review and personalize generated content before submission.