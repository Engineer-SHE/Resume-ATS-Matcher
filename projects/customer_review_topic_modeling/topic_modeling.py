"""Customer review topic modelling demo.

This script loads a small set of example reviews and trains a Latent Dirichlet Allocation
model to discover topics such as shipping issues, product quality, and service experience.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


DATA_PATH = Path(__file__).resolve().parent / "data" / "sample_reviews.csv"


@dataclass
class TopicModelConfig:
    """Configuration for the topic modelling pipeline."""

    num_topics: int = 4
    max_features: int = 1000
    max_df: float = 0.9
    min_df: int = 2
    n_top_words: int = 10
    random_state: int = 42


class ReviewTopicModeler:
    """Wraps vectorisation, LDA training, and topic interpretation."""

    def __init__(self, config: TopicModelConfig) -> None:
        self.config = config
        self.vectorizer = CountVectorizer(
            stop_words="english",
            max_df=config.max_df,
            min_df=config.min_df,
            max_features=config.max_features,
        )
        self.model = LatentDirichletAllocation(
            n_components=config.num_topics,
            learning_method="batch",
            random_state=config.random_state,
        )
        self._feature_names: List[str] | None = None
        self._topic_word_matrix: np.ndarray | None = None

    def fit_transform(self, documents: Iterable[str]) -> np.ndarray:
        """Train the topic model and return topic probabilities for each document."""

        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self._feature_names = self.vectorizer.get_feature_names_out().tolist()
        topic_distribution = self.model.fit_transform(doc_term_matrix)
        self._topic_word_matrix = self.model.components_
        return topic_distribution

    def top_words_per_topic(self) -> Dict[int, List[str]]:
        """Return the most representative words for each topic."""

        if self._feature_names is None or self._topic_word_matrix is None:
            raise RuntimeError("Model must be fitted before extracting topics.")

        top_words: Dict[int, List[str]] = {}
        for topic_idx, topic_weights in enumerate(self._topic_word_matrix):
            top_feature_indices = topic_weights.argsort()[::-1][: self.config.n_top_words]
            top_words[topic_idx] = [self._feature_names[i] for i in top_feature_indices]
        return top_words

    @staticmethod
    def label_documents(topic_distribution: np.ndarray) -> List[int]:
        """Assign the most probable topic to each document."""

        return topic_distribution.argmax(axis=1).tolist()


def load_reviews(path: Path) -> pd.DataFrame:
    """Load the review dataset from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    if "review_text" not in df.columns:
        raise ValueError("Expected a 'review_text' column in the dataset")

    df["review_text"] = df["review_text"].fillna("")
    return df


def summarise_topics(df: pd.DataFrame, topic_labels: List[int]) -> pd.DataFrame:
    """Aggregate review counts per topic and rating bucket."""

    summary = (
        df.assign(topic=topic_labels)
        .groupby(["topic", "rating"], as_index=False)
        .size()
        .pivot(index="topic", columns="rating", values="size")
        .fillna(0)
        .astype(int)
    )
    summary.columns = [f"rating_{col}" for col in summary.columns]
    summary["total_reviews"] = summary.sum(axis=1)
    return summary.sort_index()


def main() -> None:
    config = TopicModelConfig()
    reviews = load_reviews(DATA_PATH)

    modeler = ReviewTopicModeler(config)
    topic_distribution = modeler.fit_transform(reviews["review_text"].tolist())
    topic_labels = ReviewTopicModeler.label_documents(topic_distribution)

    topic_keywords = modeler.top_words_per_topic()
    topic_summary = summarise_topics(reviews, topic_labels)

    print("\n=== Topic keywords ===")
    for topic_idx, keywords in topic_keywords.items():
        print(f"Topic {topic_idx}: {', '.join(keywords)}")

    labelled_reviews = reviews.assign(topic=topic_labels, topic_probability=topic_distribution.max(axis=1))
    print("\n=== Sample labelled reviews ===")
    sample_columns = ["review_id", "source", "rating", "topic", "topic_probability", "review_text"]
    print(labelled_reviews[sample_columns].head(10).to_string(index=False))

    print("\n=== Topic distribution by rating ===")
    print(topic_summary.to_string())


if __name__ == "__main__":
    main()
