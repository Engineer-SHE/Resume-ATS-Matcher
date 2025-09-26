import streamlit as st
import tempfile
from pathlib import Path
import re
from collections import Counter
import PyPDF2
from docx import Document
import pandas as pd
import plotly.graph_objects as go
from rake_nltk import Rake
import nltk

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

st.set_page_config(
    page_title="Resume-to-Job ATS Matcher (Lite)",
    page_icon="📄",
    layout="wide"
)

st.title("🎯 Resume-to-Job Matching System (Lite Version)")
st.info("This is a lightweight version that works without heavy ML dependencies. It provides keyword-based matching and basic ATS optimization.")

class SimpleTextExtractor:
    @staticmethod
    def extract_from_file(file_path):
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension == '.pdf':
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        elif extension in ['.docx', '.doc']:
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif extension == '.txt':
            return file_path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {extension}")

class SimpleKeywordAnalyzer:
    def __init__(self):
        try:
            self.rake = Rake(min_length=1, max_length=2)  # Limit to 1-2 word phrases
        except:
            # If RAKE fails, download required data and try again
            import nltk
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            self.rake = Rake(min_length=1, max_length=2)

        # Define technology and skill keywords
        self.tech_keywords = {
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'go', 'rust', 'kotlin', 'swift', 'php', 'scala', 'r',

            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'spring', 'rails', 'laravel',

            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'dynamodb', 'elasticsearch', 'oracle', 'sqlite',

            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'ci/cd', 'devops', 'cloud',

            # Data & AI
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'nlp', 'ai',

            # Tools & Frameworks
            'git', 'github', 'gitlab', 'jira', 'confluence', 'slack', 'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum',

            # Skills
            'leadership', 'communication', 'teamwork', 'problem-solving', 'analytical', 'testing', 'debugging', 'optimization'
        }

    def extract_keywords(self, text, top_n=20):
        text_lower = text.lower()
        keywords = []

        # Extract known technical keywords
        tech_found = []
        for keyword in self.tech_keywords:
            if keyword in text_lower:
                tech_found.append(keyword)

        # Extract additional patterns
        patterns = {
            # Programming languages and frameworks
            r'\b(?:node\.?js|next\.?js|nest\.?js|vue\.?js|express\.?js)\b': 'extract',
            r'\b(?:\w+\.js|\.net|c\+\+|c#)\b': 'extract',

            # Version patterns (e.g., Python 3.x, Java 11)
            r'\b(?:python|java|ruby|php)\s*\d+(?:\.\d+)?\b': 'extract',

            # Cloud services
            r'\b(?:ec2|s3|lambda|rds|ecs|eks|azure\s+\w+|google\s+cloud)\b': 'extract',

            # Certifications
            r'\b(?:certified|certification)\s+\w+\b': 'extract',

            # Years of experience
            r'\b\d+\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience\b': 'skip',

            # Common non-keywords to skip
            r'\b(?:responsible|duties|requirements|qualifications|benefits|equal\s+opportunity)\b': 'skip'
        }

        for pattern, action in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if action == 'extract':
                keywords.extend(matches)

        # Use RAKE for additional keywords (limited to short phrases)
        try:
            self.rake.extract_keywords_from_text(text)
            rake_keywords = self.rake.get_ranked_phrases()

            # Filter RAKE keywords to remove long phrases and non-technical terms
            for kw in rake_keywords:
                kw_lower = kw.lower()
                # Only include if it's short and looks technical
                if (len(kw.split()) <= 2 and
                    not any(skip in kw_lower for skip in ['benefits', 'office', 'employer', 'opportunity', 'committed', 'policy']) and
                    (any(tech in kw_lower for tech in self.tech_keywords) or len(kw.split()) == 1)):
                    keywords.append(kw)
        except:
            pass

        # Combine all keywords
        all_keywords = list(set(tech_found + keywords))

        # Sort by relevance (technical keywords first)
        technical = [k for k in all_keywords if k.lower() in self.tech_keywords]
        others = [k for k in all_keywords if k.lower() not in self.tech_keywords]

        final_keywords = technical + others
        return final_keywords[:top_n]

    def calculate_match_score(self, resume_keywords, job_keywords):
        resume_set = set([k.lower() for k in resume_keywords])
        job_set = set([k.lower() for k in job_keywords])

        if not job_set:
            return 0.0

        matched = resume_set.intersection(job_set)
        return len(matched) / len(job_set)

    def gap_analysis(self, resume_keywords, job_keywords):
        resume_set = set([k.lower() for k in resume_keywords])
        job_set = set([k.lower() for k in job_keywords])

        matched = resume_set.intersection(job_set)
        missing = job_set - resume_set
        extra = resume_set - job_set

        return {
            'matched_keywords': list(matched),
            'missing_keywords': list(missing),
            'extra_keywords': list(extra),
            'coverage_score': len(matched) / len(job_set) if job_set else 0
        }

class SimpleOptimizer:
    @staticmethod
    def optimize_resume(resume_text, missing_keywords):
        optimized = resume_text

        # Filter missing keywords to only include technical/relevant ones
        tech_keywords = {
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'nodejs', 'sql', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
            'api', 'rest', 'graphql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'ci/cd', 'agile', 'scrum', 'jira', 'testing', 'debugging'
        }

        relevant_missing = [kw for kw in missing_keywords
                          if any(tech in kw.lower() for tech in tech_keywords)
                          or len(kw.split()) == 1][:10]

        # Add a skills section with relevant missing keywords if not present
        if relevant_missing and 'skills' not in resume_text.lower():
            skills_section = "\n\nSKILLS\n" + ", ".join(relevant_missing)
            optimized += skills_section

        # Enhance with action verbs if missing
        action_verbs = ['developed', 'implemented', 'designed', 'managed', 'created', 'built', 'optimized', 'led', 'delivered']
        verbs_added = 0
        for verb in action_verbs:
            if verb not in optimized.lower() and verbs_added < 3:
                # Add to bullet points or paragraphs
                lines = optimized.split('\n')
                for i, line in enumerate(lines):
                    if (line.strip().startswith('-') or line.strip().startswith('•')) and verb not in line.lower():
                        lines[i] = line.replace('-', f'- {verb.capitalize()}', 1).replace('•', f'• {verb.capitalize()}', 1)
                        verbs_added += 1
                        break
                optimized = '\n'.join(lines)

        return optimized

def create_match_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': "Match Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_keyword_pie(gap_analysis):
    labels = ['Matched', 'Missing', 'Extra']
    values = [
        len(gap_analysis['matched_keywords']),
        len(gap_analysis['missing_keywords']),
        len(gap_analysis['extra_keywords'])
    ]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title="Keyword Coverage", height=400)
    return fig

def main():
    extractor = SimpleTextExtractor()
    analyzer = SimpleKeywordAnalyzer()
    optimizer = SimpleOptimizer()

    with st.sidebar:
        st.header("📁 Upload Documents")

        resume_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'docx', 'txt']
        )

        job_file = st.file_uploader(
            "Upload Job Description",
            type=['pdf', 'docx', 'txt']
        )

        st.divider()

        top_keywords = st.slider(
            "Keywords to Extract",
            min_value=10,
            max_value=30,
            value=20
        )

    if resume_file and job_file:
        with st.spinner("Processing..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save uploaded files
                resume_path = Path(tmpdir) / resume_file.name
                job_path = Path(tmpdir) / job_file.name

                resume_path.write_bytes(resume_file.getbuffer())
                job_path.write_bytes(job_file.getbuffer())

                try:
                    # Extract text
                    resume_text = extractor.extract_from_file(str(resume_path))
                    job_text = extractor.extract_from_file(str(job_path))

                    # Extract keywords
                    resume_keywords = analyzer.extract_keywords(resume_text, top_keywords)
                    job_keywords = analyzer.extract_keywords(job_text, top_keywords)

                    # Calculate match
                    match_score = analyzer.calculate_match_score(resume_keywords, job_keywords)
                    gap_analysis = analyzer.gap_analysis(resume_keywords, job_keywords)

                    # Display results
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Match Score", f"{match_score * 100:.1f}%")

                    with col2:
                        st.metric("Matched Keywords", len(gap_analysis['matched_keywords']))

                    with col3:
                        st.metric("Missing Keywords", len(gap_analysis['missing_keywords']))

                    st.divider()

                    # Visualizations
                    col1, col2 = st.columns(2)

                    with col1:
                        fig = create_match_gauge(match_score)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = create_keyword_pie(gap_analysis)
                        st.plotly_chart(fig, use_container_width=True)

                    # Keyword Analysis
                    st.subheader("📊 Keyword Analysis")

                    col1, col2 = st.columns(2)

                    with col1:
                        with st.expander("✅ Matched Keywords"):
                            if gap_analysis['matched_keywords']:
                                for kw in gap_analysis['matched_keywords'][:10]:
                                    st.write(f"• {kw}")

                    with col2:
                        with st.expander("❌ Missing Keywords (Add these!)"):
                            if gap_analysis['missing_keywords']:
                                for kw in gap_analysis['missing_keywords'][:10]:
                                    st.write(f"• {kw}")

                    # Optimization
                    st.subheader("🚀 Resume Optimization")

                    if st.button("Generate Optimized Resume"):
                        optimized = optimizer.optimize_resume(
                            resume_text,
                            gap_analysis['missing_keywords']
                        )

                        st.text_area(
                            "Optimized Resume",
                            value=optimized,
                            height=400
                        )

                        st.download_button(
                            label="📥 Download Optimized Resume",
                            data=optimized,
                            file_name="optimized_resume.txt",
                            mime="text/plain"
                        )

                        st.success("✅ Resume optimized with missing keywords!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    else:
        st.info("👆 Please upload both your resume and job description to start!")

        with st.expander("ℹ️ About Lite Version"):
            st.markdown("""
            This lite version provides:
            - Keyword extraction and matching
            - Gap analysis
            - Basic ATS optimization
            - Visual analytics

            For advanced features like semantic matching and AI-powered optimization,
            please install the full version with ML dependencies.
            """)

if __name__ == "__main__":
    main()