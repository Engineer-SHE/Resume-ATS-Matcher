import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from .config import Config

logger = logging.getLogger(__name__)

class SemanticMatcher:
    def __init__(self, model_name: str = Config.MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.section_weights = Config.SECTION_WEIGHTS

    def encode_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text, convert_to_numpy=True)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0

        embedding1 = self.encode_text(text1)
        embedding2 = self.encode_text(text2)

        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0, 0]

        return float(similarity)

    def match_resume_to_job(
        self,
        resume_sections: Dict[str, str],
        job_sections: Dict[str, str]
    ) -> Dict[str, Any]:
        overall_match = 0.0
        section_scores = {}
        detailed_analysis = {}

        for section, weight in self.section_weights.items():
            resume_text = resume_sections.get(section, '')
            job_text = job_sections.get('full_text', '')

            if resume_text and job_text:
                score = self.calculate_similarity(resume_text, job_text)
                section_scores[section] = score
                overall_match += score * weight
                detailed_analysis[section] = {
                    'score': score,
                    'weight': weight,
                    'contribution': score * weight
                }
            else:
                section_scores[section] = 0.0
                detailed_analysis[section] = {
                    'score': 0.0,
                    'weight': weight,
                    'contribution': 0.0
                }

        full_text_similarity = self.calculate_similarity(
            resume_sections.get('full_text', ''),
            job_sections.get('full_text', '')
        )

        return {
            'overall_match': min(overall_match, 1.0),
            'full_text_similarity': full_text_similarity,
            'section_scores': section_scores,
            'detailed_analysis': detailed_analysis,
            'recommendation': self._generate_recommendation(overall_match)
        }

    def _generate_recommendation(self, score: float) -> str:
        if score >= 0.85:
            return "Excellent match! Your resume strongly aligns with the job requirements."
        elif score >= 0.70:
            return "Good match! Consider highlighting more relevant keywords and experiences."
        elif score >= 0.50:
            return "Moderate match. Significant improvements needed in skills and experience alignment."
        else:
            return "Low match. Major revisions recommended to better align with job requirements."

    def find_semantic_gaps(
        self,
        resume_text: str,
        job_requirements: List[str]
    ) -> List[Tuple[str, float]]:
        gaps = []
        resume_embedding = self.encode_text(resume_text)

        for requirement in job_requirements:
            req_embedding = self.encode_text(requirement)
            similarity = cosine_similarity(
                resume_embedding.reshape(1, -1),
                req_embedding.reshape(1, -1)
            )[0, 0]

            if similarity < Config.ATS_KEYWORDS_THRESHOLD:
                gaps.append((requirement, float(similarity)))

        return sorted(gaps, key=lambda x: x[1])

    def rank_keywords_by_relevance(
        self,
        keywords: List[str],
        target_text: str
    ) -> List[Tuple[str, float]]:
        target_embedding = self.encode_text(target_text)
        ranked_keywords = []

        for keyword in keywords:
            keyword_embedding = self.encode_text(keyword)
            relevance = cosine_similarity(
                keyword_embedding.reshape(1, -1),
                target_embedding.reshape(1, -1)
            )[0, 0]
            ranked_keywords.append((keyword, float(relevance)))

        return sorted(ranked_keywords, key=lambda x: x[1], reverse=True)