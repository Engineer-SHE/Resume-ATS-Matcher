import re
from typing import Dict, List, Set, Tuple
from collections import Counter
import nltk
from rake_nltk import Rake
import spacy
from .config import Config
import logging

logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    def __init__(self):
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass

        try:
            self.nlp = spacy.load(Config.SPACY_MODEL)
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", Config.SPACY_MODEL])
            self.nlp = spacy.load(Config.SPACY_MODEL)

        self.rake = Rake()
        self.skill_categories = Config.SKILL_CATEGORIES

    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        text = text.lower()

        self.rake.extract_keywords_from_text(text)
        rake_keywords = self.rake.get_ranked_phrases()[:top_n]

        doc = self.nlp(text)
        entities = [ent.text.lower() for ent in doc.ents if ent.label_ in
                   ['ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART']]

        technical_pattern = r'\b(?:' + '|'.join([
            r'python', r'java\b', r'javascript', r'react', r'angular', r'vue',
            r'node\.?js', r'sql', r'nosql', r'mongodb', r'postgresql', r'mysql',
            r'aws', r'azure', r'gcp', r'docker', r'kubernetes', r'git',
            r'ci/cd', r'devops', r'machine learning', r'deep learning',
            r'data science', r'artificial intelligence', r'blockchain',
            r'cloud computing', r'microservices', r'rest api', r'graphql'
        ]) + r')\b'

        technical_keywords = re.findall(technical_pattern, text, re.IGNORECASE)

        all_keywords = set(rake_keywords[:10] + entities + technical_keywords)
        return list(all_keywords)[:top_n]

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        text_lower = text.lower()
        extracted_skills = {
            'technical': [],
            'soft': [],
            'action_verbs': []
        }

        for category, skills in self.skill_categories.items():
            for skill in skills:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    extracted_skills[category].append(skill)

        additional_technical = self._extract_technical_skills(text)
        extracted_skills['technical'].extend(additional_technical)
        extracted_skills['technical'] = list(set(extracted_skills['technical']))

        return extracted_skills

    def _extract_technical_skills(self, text: str) -> List[str]:
        patterns = {
            'programming': r'\b(python|java|c\+\+|c#|javascript|typescript|ruby|go|rust|scala|kotlin)\b',
            'frameworks': r'\b(django|flask|spring|express|react|angular|vue|svelte|next\.?js)\b',
            'databases': r'\b(mysql|postgresql|mongodb|redis|cassandra|dynamodb|elasticsearch)\b',
            'cloud': r'\b(aws|azure|gcp|google cloud|amazon web services|microsoft azure)\b',
            'tools': r'\b(docker|kubernetes|jenkins|terraform|ansible|puppet|chef)\b',
            'data': r'\b(pandas|numpy|scikit-learn|tensorflow|pytorch|keras|tableau|powerbi)\b'
        }

        skills = []
        text_lower = text.lower()

        for category, pattern in patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            skills.extend(matches)

        return list(set(skills))

    def gap_analysis(
        self,
        resume_keywords: List[str],
        job_keywords: List[str]
    ) -> Dict[str, Any]:
        resume_set = set([k.lower() for k in resume_keywords])
        job_set = set([k.lower() for k in job_keywords])

        matched_keywords = resume_set.intersection(job_set)
        missing_keywords = job_set - resume_set
        extra_keywords = resume_set - job_set

        coverage_score = len(matched_keywords) / len(job_set) if job_set else 0

        return {
            'matched_keywords': list(matched_keywords),
            'missing_keywords': list(missing_keywords),
            'extra_keywords': list(extra_keywords),
            'coverage_score': coverage_score,
            'total_job_keywords': len(job_set),
            'total_resume_keywords': len(resume_set),
            'matched_count': len(matched_keywords)
        }

    def analyze_action_verbs(self, text: str) -> Dict[str, Any]:
        action_verbs = [
            'achieved', 'analyzed', 'built', 'collaborated', 'created', 'designed',
            'developed', 'enhanced', 'established', 'executed', 'implemented',
            'improved', 'increased', 'initiated', 'launched', 'led', 'managed',
            'optimized', 'organized', 'pioneered', 'reduced', 'resolved', 'streamlined',
            'supervised', 'transformed'
        ]

        text_lower = text.lower()
        found_verbs = []
        verb_counts = {}

        for verb in action_verbs:
            pattern = r'\b' + verb + r'(?:d|ed|ing)?\b'
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                found_verbs.append(verb)
                verb_counts[verb] = len(matches)

        sentences = text.split('.')
        action_sentences = []
        for sentence in sentences:
            for verb in found_verbs:
                if verb in sentence.lower():
                    action_sentences.append(sentence.strip())
                    break

        return {
            'found_action_verbs': found_verbs,
            'verb_counts': verb_counts,
            'total_action_verbs': len(found_verbs),
            'action_verb_density': len(found_verbs) / len(action_verbs),
            'sample_action_sentences': action_sentences[:5]
        }

    def keyword_density_analysis(self, text: str, keywords: List[str]) -> Dict[str, float]:
        text_lower = text.lower()
        word_count = len(text_lower.split())
        density_map = {}

        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = text_lower.count(keyword_lower)
            density = (count / word_count) * 100 if word_count > 0 else 0
            density_map[keyword] = round(density, 2)

        return dict(sorted(density_map.items(), key=lambda x: x[1], reverse=True))

    def extract_quantifiable_achievements(self, text: str) -> List[str]:
        patterns = [
            r'(?:increased|improved|reduced|saved|generated|grew|expanded).*?\d+[%$]',
            r'\d+[%$].*?(?:increase|improvement|reduction|savings|growth)',
            r'(?:managed|led|supervised).*?\d+.*?(?:team|people|members|employees)',
            r'\$[\d,]+(?:K|M|B)?',
            r'\d+\+?\s*(?:years?|months?)\s*(?:of\s*)?experience'
        ]

        achievements = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            achievements.extend(matches)

        return list(set(achievements))