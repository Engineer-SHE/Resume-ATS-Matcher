import streamlit as st
import tempfile
import os
from pathlib import Path
import pandas as pd
import json
from src.text_extractor import TextExtractor
from src.semantic_matcher import SemanticMatcher
from src.keyword_analyzer import KeywordAnalyzer
from src.ats_optimizer import ATSOptimizer
from src.visualizer import Visualizer
from src.config import Config

st.set_page_config(
    page_title="Resume-to-Job ATS Matcher",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def initialize_components():
    Config.setup_directories()
    return {
        'text_extractor': TextExtractor(),
        'semantic_matcher': SemanticMatcher(),
        'keyword_analyzer': KeywordAnalyzer(),
        'ats_optimizer': ATSOptimizer(),
        'visualizer': Visualizer()
    }

def main():
    st.title("🎯 Resume-to-Job Matching System with ATS Mode")
    st.markdown("""
    This AI-powered tool analyzes how well your resume matches a job description and provides
    an ATS-optimized version to improve your chances of getting noticed.
    """)

    components = initialize_components()

    with st.sidebar:
        st.header("📁 Upload Documents")

        resume_file = st.file_uploader(
            "Upload Resume",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume in PDF, DOCX, or TXT format"
        )

        job_file = st.file_uploader(
            "Upload Job Description",
            type=['pdf', 'docx', 'txt'],
            help="Upload the job description you're applying for"
        )

        st.divider()

        st.header("⚙️ Settings")

        analysis_mode = st.radio(
            "Analysis Mode",
            ["Quick Match", "Detailed Analysis", "ATS Optimization"],
            index=1
        )

        top_keywords = st.slider(
            "Number of Keywords to Extract",
            min_value=10,
            max_value=50,
            value=20
        )

    if resume_file and job_file:
        with st.spinner("Processing documents..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                resume_path = Path(tmpdir) / resume_file.name
                job_path = Path(tmpdir) / job_file.name

                with open(resume_path, 'wb') as f:
                    f.write(resume_file.getbuffer())
                with open(job_path, 'wb') as f:
                    f.write(job_file.getbuffer())

                try:
                    resume_text = components['text_extractor'].extract_from_file(str(resume_path))
                    job_text = components['text_extractor'].extract_from_file(str(job_path))

                    resume_sections = components['text_extractor'].extract_sections(resume_text)
                    job_sections = {'full_text': job_text}

                    resume_keywords = components['keyword_analyzer'].extract_keywords(resume_text, top_keywords)
                    job_keywords = components['keyword_analyzer'].extract_keywords(job_text, top_keywords)

                    if analysis_mode == "Quick Match":
                        display_quick_match(
                            components,
                            resume_sections,
                            job_sections,
                            resume_keywords,
                            job_keywords
                        )

                    elif analysis_mode == "Detailed Analysis":
                        display_detailed_analysis(
                            components,
                            resume_sections,
                            job_sections,
                            resume_keywords,
                            job_keywords
                        )

                    elif analysis_mode == "ATS Optimization":
                        display_ats_optimization(
                            components,
                            resume_sections,
                            job_text,
                            job_keywords
                        )

                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")

    else:
        st.info("👆 Please upload both your resume and a job description to get started!")

        with st.expander("📚 How to Use This Tool"):
            st.markdown("""
            1. **Upload Your Resume**: Click on 'Upload Resume' in the sidebar
            2. **Upload Job Description**: Upload the job posting you're interested in
            3. **Choose Analysis Mode**:
               - **Quick Match**: Get a fast compatibility score
               - **Detailed Analysis**: Deep dive into gaps and improvements
               - **ATS Optimization**: Generate an ATS-friendly version of your resume
            4. **Review Results**: Analyze the match score, missing keywords, and suggestions
            5. **Download Optimized Resume**: Get your improved resume (in ATS mode)
            """)

def display_quick_match(components, resume_sections, job_sections, resume_keywords, job_keywords):
    st.header("⚡ Quick Match Results")

    match_results = components['semantic_matcher'].match_resume_to_job(
        resume_sections,
        job_sections
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Match",
            f"{match_results['overall_match'] * 100:.1f}%",
            delta=f"{(match_results['overall_match'] - 0.7) * 100:.1f}%"
        )

    with col2:
        gap_analysis = components['keyword_analyzer'].gap_analysis(resume_keywords, job_keywords)
        st.metric(
            "Keyword Coverage",
            f"{gap_analysis['coverage_score'] * 100:.1f}%",
            delta=f"{len(gap_analysis['matched_keywords'])} matched"
        )

    with col3:
        st.metric(
            "Missing Keywords",
            len(gap_analysis['missing_keywords']),
            delta=f"-{len(gap_analysis['missing_keywords'])}"
        )

    with col4:
        recommendation_color = "🟢" if match_results['overall_match'] >= 0.7 else "🔴"
        st.metric(
            "Recommendation",
            recommendation_color,
            delta=None
        )

    st.divider()

    st.subheader("📊 Match Visualization")
    fig = components['visualizer'].create_match_gauge(match_results['overall_match'])
    st.plotly_chart(fig, use_container_width=True)

    st.info(match_results['recommendation'])

def display_detailed_analysis(components, resume_sections, job_sections, resume_keywords, job_keywords):
    st.header("🔍 Detailed Analysis")

    tabs = st.tabs(["Match Analysis", "Keyword Gap", "Skills Analysis", "Action Verbs"])

    with tabs[0]:
        st.subheader("Section-wise Match Analysis")

        match_results = components['semantic_matcher'].match_resume_to_job(
            resume_sections,
            job_sections
        )

        col1, col2 = st.columns([1, 2])

        with col1:
            fig = components['visualizer'].create_match_gauge(match_results['overall_match'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = components['visualizer'].create_section_scores_chart(match_results['section_scores'])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Section Contribution Breakdown")
        df = pd.DataFrame(match_results['detailed_analysis']).T
        df['contribution'] = df['contribution'] * 100
        df = df.round(2)
        st.dataframe(df, use_container_width=True)

    with tabs[1]:
        st.subheader("Keyword Gap Analysis")

        gap_analysis = components['keyword_analyzer'].gap_analysis(resume_keywords, job_keywords)

        col1, col2 = st.columns(2)

        with col1:
            fig = components['visualizer'].create_keyword_coverage_chart(gap_analysis)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Coverage Score", f"{gap_analysis['coverage_score'] * 100:.1f}%")
            st.metric("Matched Keywords", gap_analysis['matched_count'])
            st.metric("Missing Keywords", len(gap_analysis['missing_keywords']))

        with st.expander("🔴 Missing Keywords (Add these to your resume)"):
            if gap_analysis['missing_keywords']:
                missing_df = pd.DataFrame({'Missing Keywords': gap_analysis['missing_keywords']})
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing keywords!")

        with st.expander("🟢 Matched Keywords"):
            if gap_analysis['matched_keywords']:
                matched_df = pd.DataFrame({'Matched Keywords': gap_analysis['matched_keywords']})
                st.dataframe(matched_df, use_container_width=True)

    with tabs[2]:
        st.subheader("Skills Analysis")

        resume_skills = components['keyword_analyzer'].extract_skills(resume_sections['full_text'])
        job_skills = components['keyword_analyzer'].extract_skills(job_sections['full_text'])

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Your Skills**")
            for category, skills in resume_skills.items():
                if skills:
                    st.write(f"**{category.title()}:** {', '.join(skills)}")

        with col2:
            st.write("**Required Skills**")
            for category, skills in job_skills.items():
                if skills:
                    st.write(f"**{category.title()}:** {', '.join(skills)}")

        fig = components['visualizer'].create_skills_radar_chart(resume_skills)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Action Verbs Analysis")

        action_analysis = components['keyword_analyzer'].analyze_action_verbs(
            resume_sections['full_text']
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Action Verbs Found", action_analysis['total_action_verbs'])

        with col2:
            st.metric("Action Verb Density", f"{action_analysis['action_verb_density'] * 100:.1f}%")

        with col3:
            st.metric("Unique Action Verbs", len(action_analysis['found_action_verbs']))

        fig = components['visualizer'].create_action_verbs_chart(action_analysis)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Sample Action Sentences"):
            for sentence in action_analysis['sample_action_sentences']:
                st.write(f"• {sentence}")

def display_ats_optimization(components, resume_sections, job_text, job_keywords):
    st.header("🚀 ATS Optimization")

    with st.spinner("Optimizing your resume for ATS..."):
        optimization_results = components['ats_optimizer'].optimize_resume(
            resume_sections,
            job_text,
            job_keywords
        )

    tabs = st.tabs(["Optimization Results", "Before vs After", "Optimized Resume", "Download"])

    with tabs[0]:
        st.subheader("Optimization Summary")

        improvement = optimization_results['improvement_score']
        improvement_pct = improvement * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Overall Improvement",
                f"{improvement_pct:.1f}%",
                delta=f"+{improvement_pct:.1f}%"
            )

        with col2:
            st.metric(
                "Keyword Coverage",
                f"{optimization_results['metrics_after']['keyword_coverage'] * 100:.1f}%",
                delta=f"+{(optimization_results['metrics_after']['keyword_coverage'] - optimization_results['metrics_before']['keyword_coverage']) * 100:.1f}%"
            )

        with col3:
            st.metric(
                "Action Verbs",
                optimization_results['metrics_after']['action_verbs_count'],
                delta=f"+{optimization_results['metrics_after']['action_verbs_count'] - optimization_results['metrics_before']['action_verbs_count']}"
            )

        st.subheader("Optimization Suggestions Applied")
        for section, suggestions in optimization_results['suggestions'].items():
            if suggestions:
                with st.expander(f"📝 {section.title()} Section"):
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")

    with tabs[1]:
        st.subheader("Before vs After Comparison")

        fig = components['visualizer'].create_before_after_comparison(
            optimization_results['metrics_before'],
            optimization_results['metrics_after']
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Before Optimization**")
            st.json(optimization_results['metrics_before'])

        with col2:
            st.write("**After Optimization**")
            st.json(optimization_results['metrics_after'])

    with tabs[2]:
        st.subheader("Optimized Resume Preview")

        for section_name, section_text in optimization_results['optimized_resume'].items():
            if section_name != 'full_text' and section_text:
                st.write(f"**{section_name.upper()}**")
                st.text_area(
                    "",
                    value=section_text,
                    height=150,
                    key=f"optimized_{section_name}"
                )

    with tabs[3]:
        st.subheader("Download Optimized Resume")

        optimized_text = optimization_results['optimized_resume']['full_text']

        st.download_button(
            label="📥 Download Optimized Resume (TXT)",
            data=optimized_text,
            file_name="optimized_resume.txt",
            mime="text/plain"
        )

        optimization_report = {
            'metrics_before': optimization_results['metrics_before'],
            'metrics_after': optimization_results['metrics_after'],
            'improvement_score': optimization_results['improvement_score'],
            'suggestions': optimization_results['suggestions']
        }

        st.download_button(
            label="📊 Download Optimization Report (JSON)",
            data=json.dumps(optimization_report, indent=2),
            file_name="optimization_report.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()