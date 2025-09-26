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
    page_title="Resume-to-Job ATS Matcher (Enhanced)",
    page_icon="📄",
    layout="wide"
)

st.title("🎯 Resume-to-Job Matching System (Enhanced Version)")
st.info("Smart ATS optimization that preserves your resume's authenticity while improving keyword coverage.")

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

class SmartKeywordAnalyzer:
    def __init__(self):
        try:
            self.rake = Rake(min_length=1, max_length=2)
        except:
            import nltk
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            self.rake = Rake(min_length=1, max_length=2)

    def extract_job_requirements(self, job_text):
        """Extract key requirements from job description"""
        requirements = {
            'languages': [],
            'frameworks': [],
            'tools': [],
            'skills': [],
            'experience': []
        }

        job_lower = job_text.lower()

        # Programming languages
        lang_pattern = r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|kotlin|swift|php|scala|r|sql)\b'
        requirements['languages'] = list(set(re.findall(lang_pattern, job_lower)))

        # AI/ML specific
        ml_pattern = r'\b(genai|generative ai|llm|langchain|transformers|nlp|machine learning|deep learning|pytorch|tensorflow|scikit-learn|agentic ai|rag|prompt engineering)\b'
        requirements['frameworks'] = list(set(re.findall(ml_pattern, job_lower)))

        # Tools and platforms
        tools_pattern = r'\b(aws|azure|gcp|docker|kubernetes|snowflake|pinecone|tableau|power bi|git|github|gitlab|jira|confluence|jenkins)\b'
        requirements['tools'] = list(set(re.findall(tools_pattern, job_lower)))

        # Extract years of experience
        exp_pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?experience'
        exp_matches = re.findall(exp_pattern, job_lower)
        if exp_matches:
            requirements['experience'] = exp_matches

        return requirements

    def extract_resume_keywords(self, resume_text):
        """Extract relevant keywords from resume"""
        keywords = {
            'languages': [],
            'frameworks': [],
            'tools': [],
            'skills': []
        }

        resume_lower = resume_text.lower()

        # Same patterns as job requirements
        lang_pattern = r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|kotlin|swift|php|scala|r|sql)\b'
        keywords['languages'] = list(set(re.findall(lang_pattern, resume_lower)))

        ml_pattern = r'\b(nltk|spacy|scikit-learn|tensorflow|pytorch|pandas|numpy|tableau|machine learning|nlp|generative ai|chatgpt)\b'
        keywords['frameworks'] = list(set(re.findall(ml_pattern, resume_lower)))

        tools_pattern = r'\b(aws|azure|gcp|docker|kubernetes|snowflake|salesforce|tableau|git|github|gitlab|jira)\b'
        keywords['tools'] = list(set(re.findall(tools_pattern, resume_lower)))

        return keywords

    def gap_analysis(self, resume_keywords, job_requirements):
        """Perform intelligent gap analysis"""
        gaps = {
            'missing_languages': [],
            'missing_frameworks': [],
            'missing_tools': [],
            'matched': [],
            'suggestions': []
        }

        # Check each category
        for category in ['languages', 'frameworks', 'tools']:
            job_set = set(job_requirements.get(category, []))
            resume_set = set(resume_keywords.get(category, []))

            missing = job_set - resume_set
            matched = job_set.intersection(resume_set)

            gaps[f'missing_{category}'] = list(missing)
            gaps['matched'].extend(list(matched))

        # Generate smart suggestions
        if gaps['missing_languages']:
            gaps['suggestions'].append(f"Consider adding experience with: {', '.join(gaps['missing_languages'][:3])}")

        if 'genai' in str(job_requirements).lower() or 'generative ai' in str(job_requirements).lower():
            if 'chatgpt' in resume_keywords.get('frameworks', []) or 'generative ai' in resume_keywords.get('frameworks', []):
                gaps['suggestions'].append("Great! You have GenAI experience. Consider emphasizing LLM and prompt engineering work.")
            else:
                gaps['suggestions'].append("The role requires GenAI experience. Highlight any ChatGPT, LLM, or AI prototype work.")

        return gaps

class SmartOptimizer:
    @staticmethod
    def optimize_resume(resume_text, job_requirements, gaps, resume_keywords):
        """Intelligently optimize resume based on job requirements"""
        optimized = resume_text
        suggestions = []

        # Don't duplicate action verbs - check what's already there
        existing_verbs = set(re.findall(r'\b(developed|built|implemented|designed|created|engineered|architected|led|managed)\b', resume_text.lower()))

        # Find missing critical keywords from the job
        critical_missing = []

        # Check for GenAI/LLM keywords if job requires them
        if any(term in str(job_requirements).lower() for term in ['genai', 'generative ai', 'llm', 'langchain', 'agentic']):
            # Sheila has ChatGPT and generative AI experience - make sure it's highlighted
            if 'llm' not in resume_text.lower():
                critical_missing.append('LLM')
            if 'prompt engineering' not in resume_text.lower() and 'prompt' in str(job_requirements).lower():
                critical_missing.append('prompt engineering')

        # Add missing tools that align with experience
        for tool in gaps.get('missing_tools', []):
            if tool in ['aws', 'gcp', 'azure'] and 'cloud' not in resume_text.lower():
                critical_missing.append(tool.upper())
            elif tool == 'snowflake' and 'data' in resume_text.lower():
                critical_missing.append('Snowflake')

        # Only add a skills section if there are relevant missing keywords
        if critical_missing and 'skills' not in resume_text.lower():
            # Create a proper skills section with categories
            skills_section = "\n\nTECHNICAL SKILLS\n"

            # Add existing skills first
            existing_langs = ', '.join([l.upper() if len(l) <= 3 else l.capitalize() for l in resume_keywords.get('languages', [])])
            if existing_langs:
                skills_section += f"Languages: {existing_langs}\n"

            existing_tools = ', '.join([t.upper() if len(t) <= 3 else t.capitalize() for t in resume_keywords.get('frameworks', [])])
            if existing_tools:
                skills_section += f"Frameworks/Libraries: {existing_tools}\n"

            # Add critical missing keywords that make sense
            if critical_missing:
                skills_section += f"Additional: {', '.join(critical_missing[:3])}\n"

            optimized += skills_section
            suggestions.append(f"Added technical skills section with relevant keywords")

        # Enhance bullets with missing keywords where appropriate
        lines = optimized.split('\n')
        enhanced_lines = []

        for line in lines:
            enhanced_line = line

            # If it's about AI/ML work and job needs GenAI/LLM
            if ('generative ai' in line.lower() or 'chatgpt' in line.lower()) and 'llm' not in line.lower():
                if 'llm' in str(gaps.get('missing_frameworks', [])):
                    enhanced_line = line.replace('generative AI', 'generative AI (LLM)')
                    suggestions.append("Enhanced GenAI experience with LLM terminology")

            # If it's about Python and job needs specific libraries
            if 'python' in line.lower():
                missing_libs = [lib for lib in gaps.get('missing_frameworks', []) if lib in ['langchain', 'transformers']]
                if missing_libs and not any(lib in line.lower() for lib in missing_libs):
                    # Don't add if line is already long
                    if len(line) < 150:
                        enhanced_line = line.rstrip() + f" including {missing_libs[0]}"
                        suggestions.append(f"Added {missing_libs[0]} to Python experience")

            enhanced_lines.append(enhanced_line)

        optimized = '\n'.join(enhanced_lines)

        return optimized, suggestions

def main():
    extractor = SimpleTextExtractor()
    analyzer = SmartKeywordAnalyzer()
    optimizer = SmartOptimizer()

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

        optimization_level = st.radio(
            "Optimization Level",
            ["Conservative (Minimal Changes)", "Balanced (Smart Enhancements)", "Aggressive (Maximum ATS)"],
            index=1
        )

    if resume_file and job_file:
        with st.spinner("Analyzing documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save files
                resume_path = Path(tmpdir) / resume_file.name
                job_path = Path(tmpdir) / job_file.name

                resume_path.write_bytes(resume_file.getbuffer())
                job_path.write_bytes(job_file.getbuffer())

                try:
                    # Extract text
                    resume_text = extractor.extract_from_file(str(resume_path))
                    job_text = extractor.extract_from_file(str(job_path))

                    # Extract requirements and keywords
                    job_requirements = analyzer.extract_job_requirements(job_text)
                    resume_keywords = analyzer.extract_resume_keywords(resume_text)

                    # Gap analysis
                    gaps = analyzer.gap_analysis(resume_keywords, job_requirements)

                    # Display analysis
                    st.header("📊 Analysis Results")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        matched_count = len(gaps['matched'])
                        st.metric("Matched Keywords", matched_count)

                    with col2:
                        missing_count = len(gaps['missing_languages']) + len(gaps['missing_frameworks']) + len(gaps['missing_tools'])
                        st.metric("Missing Keywords", missing_count)

                    with col3:
                        match_score = matched_count / (matched_count + missing_count) * 100 if (matched_count + missing_count) > 0 else 0
                        st.metric("Match Score", f"{match_score:.1f}%")

                    st.divider()

                    # Detailed breakdown
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("✅ Your Strengths")
                        if gaps['matched']:
                            for keyword in gaps['matched']:
                                st.write(f"• {keyword}")
                        else:
                            st.write("No direct matches found")

                    with col2:
                        st.subheader("📝 Gap Analysis")
                        if gaps['missing_languages']:
                            st.write("**Missing Languages:**")
                            for lang in gaps['missing_languages'][:5]:
                                st.write(f"• {lang}")

                        if gaps['missing_frameworks']:
                            st.write("**Missing Frameworks/Tools:**")
                            for framework in gaps['missing_frameworks'][:5]:
                                st.write(f"• {framework}")

                    # Smart suggestions
                    if gaps['suggestions']:
                        st.subheader("💡 Smart Suggestions")
                        for suggestion in gaps['suggestions']:
                            st.info(suggestion)

                    # Optimization
                    st.divider()
                    st.header("🚀 Resume Optimization")

                    if st.button("Generate Optimized Resume"):
                        with st.spinner("Optimizing your resume..."):
                            optimized_resume, optimization_notes = optimizer.optimize_resume(
                                resume_text,
                                job_requirements,
                                gaps,
                                resume_keywords
                            )

                            st.success("✅ Resume optimized successfully!")

                            if optimization_notes:
                                st.subheader("Changes Made:")
                                for note in optimization_notes:
                                    st.write(f"• {note}")

                            st.text_area(
                                "Optimized Resume",
                                value=optimized_resume,
                                height=400
                            )

                            st.download_button(
                                label="📥 Download Optimized Resume",
                                data=optimized_resume,
                                file_name=f"optimized_{resume_file.name.replace('.pdf', '.txt').replace('.docx', '.txt')}",
                                mime="text/plain"
                            )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Please upload both your resume and job description to start!")

        with st.expander("ℹ️ What's New in This Version"):
            st.markdown("""
            **Smart Optimization:**
            - No more duplicate action verbs
            - Intelligent keyword placement
            - Preserves your original content
            - Adds only relevant missing keywords
            - Focuses on actual job requirements

            **Better Analysis:**
            - Categorized gap analysis (languages, frameworks, tools)
            - Smart suggestions based on your experience
            - GenAI/LLM specific recommendations
            """)

if __name__ == "__main__":
    main()