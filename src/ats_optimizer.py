import re
from typing import Dict, List, Tuple, Any
from .keyword_analyzer import KeywordAnalyzer
from .semantic_matcher import SemanticMatcher
from .config import Config
import logging

logger = logging.getLogger(__name__)

class ATSOptimizer:
    def __init__(self):
        self.keyword_analyzer = KeywordAnalyzer()
        self.semantic_matcher = SemanticMatcher()

    def optimize_resume(
        self,
        resume_sections: Dict[str, str],
        job_description: str,
        job_keywords: List[str]
    ) -> Dict[str, Any]:
        optimized_sections = {}
        optimization_suggestions = {}
        metrics_before = self._calculate_metrics(resume_sections, job_keywords)

        for section_name, section_text in resume_sections.items():
            if section_name == 'full_text':
                continue

            optimized_text, suggestions = self._optimize_section(
                section_text,
                section_name,
                job_keywords,
                job_description
            )

            optimized_sections[section_name] = optimized_text
            optimization_suggestions[section_name] = suggestions

        optimized_sections['full_text'] = self._combine_sections(optimized_sections)
        metrics_after = self._calculate_metrics(optimized_sections, job_keywords)

        return {
            'original_resume': resume_sections,
            'optimized_resume': optimized_sections,
            'suggestions': optimization_suggestions,
            'metrics_before': metrics_before,
            'metrics_after': metrics_after,
            'improvement_score': self._calculate_improvement(metrics_before, metrics_after)
        }

    def _optimize_section(
        self,
        section_text: str,
        section_name: str,
        job_keywords: List[str],
        job_description: str
    ) -> Tuple[str, List[str]]:
        suggestions = []
        optimized_text = section_text

        if section_name == 'summary':
            optimized_text, sugg = self._optimize_summary(section_text, job_keywords, job_description)
            suggestions.extend(sugg)

        elif section_name == 'experience':
            optimized_text, sugg = self._optimize_experience(section_text, job_keywords)
            suggestions.extend(sugg)

        elif section_name == 'skills':
            optimized_text, sugg = self._optimize_skills(section_text, job_keywords)
            suggestions.extend(sugg)

        missing_keywords = self._find_missing_keywords(optimized_text, job_keywords)
        if missing_keywords and len(missing_keywords) <= 5:
            optimized_text, sugg = self._inject_keywords(optimized_text, missing_keywords[:3])
            suggestions.extend(sugg)

        return optimized_text, suggestions

    def _optimize_summary(
        self,
        summary: str,
        job_keywords: List[str],
        job_description: str
    ) -> Tuple[str, List[str]]:
        suggestions = []

        if not summary or len(summary) < 50:
            summary = self._generate_summary(job_description, job_keywords)
            suggestions.append("Generated new professional summary based on job requirements")

        relevant_keywords = [kw for kw in job_keywords[:5] if kw.lower() not in summary.lower()]
        if relevant_keywords:
            summary = self._enhance_with_keywords(summary, relevant_keywords[:3])
            suggestions.append(f"Added keywords: {', '.join(relevant_keywords[:3])}")

        if not self._has_action_verb_start(summary):
            summary = self._add_action_verb_start(summary)
            suggestions.append("Enhanced summary with strong action verb opening")

        return summary, suggestions

    def _optimize_experience(
        self,
        experience: str,
        job_keywords: List[str]
    ) -> Tuple[str, List[str]]:
        suggestions = []
        lines = experience.split('\n')
        optimized_lines = []

        for line in lines:
            if self._is_bullet_point(line):
                optimized_line, sugg = self._optimize_bullet_point(line, job_keywords)
                optimized_lines.append(optimized_line)
                if sugg:
                    suggestions.append(sugg)
            else:
                optimized_lines.append(line)

        optimized_text = '\n'.join(optimized_lines)

        action_verbs = self.keyword_analyzer.analyze_action_verbs(optimized_text)
        if action_verbs['total_action_verbs'] < 5:
            suggestions.append("Consider adding more action verbs to experience descriptions")

        return optimized_text, suggestions

    def _optimize_skills(
        self,
        skills: str,
        job_keywords: List[str]
    ) -> Tuple[str, List[str]]:
        suggestions = []

        current_skills = self._extract_skill_list(skills)
        job_skills = [kw for kw in job_keywords if self._is_skill_keyword(kw)]

        missing_skills = set(job_skills) - set(current_skills)
        if missing_skills:
            skills = self._add_skills(skills, list(missing_skills)[:5])
            suggestions.append(f"Added missing skills: {', '.join(list(missing_skills)[:5])}")

        skills = self._organize_skills_by_relevance(skills, job_keywords)
        suggestions.append("Reorganized skills by job relevance")

        return skills, suggestions

    def _generate_summary(self, job_description: str, keywords: List[str]) -> str:
        key_skills = ', '.join(keywords[:3])
        summary = f"Results-driven professional with expertise in {key_skills}. "
        summary += "Proven track record of delivering high-impact solutions and driving operational excellence. "
        summary += "Strong analytical and problem-solving skills with ability to adapt to dynamic environments."
        return summary

    def _enhance_with_keywords(self, text: str, keywords: List[str]) -> str:
        for keyword in keywords:
            if keyword.lower() not in text.lower():
                insertion_point = text.find('.')
                if insertion_point > 0:
                    text = text[:insertion_point] + f" with expertise in {keyword}" + text[insertion_point:]
                else:
                    text += f" Skilled in {keyword}."
        return text

    def _has_action_verb_start(self, text: str) -> bool:
        action_starters = ['accomplished', 'achieved', 'delivered', 'driven', 'experienced',
                          'innovative', 'results-oriented', 'strategic']
        first_word = text.split()[0].lower() if text else ''
        return any(starter in first_word for starter in action_starters)

    def _add_action_verb_start(self, text: str) -> str:
        if not text:
            return text
        action_starts = ['Accomplished', 'Results-driven', 'Innovative', 'Strategic']
        import random
        return f"{random.choice(action_starts)} {text[0].lower()}{text[1:]}"

    def _is_bullet_point(self, line: str) -> bool:
        return bool(re.match(r'^[\s]*[-•*]\s+', line))

    def _optimize_bullet_point(
        self,
        bullet: str,
        keywords: List[str]
    ) -> Tuple[str, str]:
        suggestion = ""

        if not re.match(r'^.*?\b(achieved|led|developed|managed|created|implemented)', bullet.lower()):
            action_verbs = ['Developed', 'Implemented', 'Led', 'Managed', 'Created']
            import random
            verb = random.choice(action_verbs)
            bullet = re.sub(r'^([\s]*[-•*]\s+)', r'\1' + verb + ' ', bullet)
            suggestion = f"Added action verb: {verb}"

        if not re.search(r'\d+[%$]?', bullet):
            if 'improved' in bullet.lower() or 'increased' in bullet.lower():
                bullet = bullet.rstrip('.') + ' by 25%.'
                suggestion += " Added quantifiable metric"

        return bullet, suggestion

    def _extract_skill_list(self, skills_text: str) -> List[str]:
        skills = re.split(r'[,;\n•*-]', skills_text)
        return [skill.strip() for skill in skills if skill.strip()]

    def _is_skill_keyword(self, keyword: str) -> bool:
        skill_indicators = ['framework', 'language', 'tool', 'platform', 'system',
                           'software', 'technology', 'methodology']
        return (len(keyword.split()) <= 3 and
                not any(word in keyword.lower() for word in ['years', 'experience', 'with']))

    def _add_skills(self, skills_text: str, new_skills: List[str]) -> str:
        if skills_text.strip():
            return skills_text.rstrip() + ', ' + ', '.join(new_skills)
        return 'Skills: ' + ', '.join(new_skills)

    def _organize_skills_by_relevance(self, skills: str, job_keywords: List[str]) -> str:
        skill_list = self._extract_skill_list(skills)
        job_kw_lower = [kw.lower() for kw in job_keywords]

        relevant_skills = []
        other_skills = []

        for skill in skill_list:
            if any(kw in skill.lower() for kw in job_kw_lower):
                relevant_skills.append(skill)
            else:
                other_skills.append(skill)

        organized = relevant_skills + other_skills
        return ', '.join(organized)

    def _find_missing_keywords(self, text: str, keywords: List[str]) -> List[str]:
        text_lower = text.lower()
        return [kw for kw in keywords if kw.lower() not in text_lower]

    def _inject_keywords(
        self,
        text: str,
        keywords: List[str]
    ) -> Tuple[str, List[str]]:
        suggestions = []
        for keyword in keywords:
            if keyword.lower() not in text.lower():
                sentences = text.split('.')
                if len(sentences) > 1:
                    mid_point = len(sentences) // 2
                    sentences[mid_point] += f" utilizing {keyword}"
                    text = '. '.join(sentences)
                    suggestions.append(f"Injected keyword: {keyword}")
        return text, suggestions

    def _combine_sections(self, sections: Dict[str, str]) -> str:
        ordered_sections = ['summary', 'experience', 'skills', 'education', 'achievements', 'projects']
        combined = []

        for section in ordered_sections:
            if section in sections and sections[section]:
                combined.append(sections[section])

        return '\n\n'.join(combined)

    def _calculate_metrics(self, resume_sections: Dict[str, str], job_keywords: List[str]) -> Dict[str, Any]:
        full_text = resume_sections.get('full_text', '')
        resume_keywords = self.keyword_analyzer.extract_keywords(full_text)

        gap_analysis = self.keyword_analyzer.gap_analysis(resume_keywords, job_keywords)
        action_verbs = self.keyword_analyzer.analyze_action_verbs(full_text)

        return {
            'keyword_coverage': gap_analysis['coverage_score'],
            'matched_keywords': len(gap_analysis['matched_keywords']),
            'missing_keywords': len(gap_analysis['missing_keywords']),
            'action_verbs_count': action_verbs['total_action_verbs'],
            'word_count': len(full_text.split())
        }

    def _calculate_improvement(self, before: Dict, after: Dict) -> float:
        improvements = []

        if before['keyword_coverage'] > 0:
            kw_improvement = (after['keyword_coverage'] - before['keyword_coverage']) / before['keyword_coverage']
            improvements.append(kw_improvement)

        if before['action_verbs_count'] > 0:
            verb_improvement = (after['action_verbs_count'] - before['action_verbs_count']) / before['action_verbs_count']
            improvements.append(verb_improvement)

        if before['missing_keywords'] > 0:
            missing_improvement = (before['missing_keywords'] - after['missing_keywords']) / before['missing_keywords']
            improvements.append(missing_improvement)

        return sum(improvements) / len(improvements) if improvements else 0.0