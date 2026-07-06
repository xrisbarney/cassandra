"""
topic_map.py
============
Maps CVE descriptions to K threat topics using NMF or LDA on TF-IDF features.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# ---------------------------------------------------------------------------
# Seed vocabulary for 8 canonical threat topics
# ---------------------------------------------------------------------------
THREAT_TOPIC_SEEDS: dict[str, list[str]] = {
    "memory_corruption": [
        "buffer overflow", "use after free", "heap spray", "stack overflow",
        "out of bounds write", "use-after-free", "double free", "heap overflow",
        "memory corruption", "stack smashing",
    ],
    "injection": [
        "sql injection", "command injection", "code injection", "xml injection",
        "ldap injection", "os command", "remote code execution", "arbitrary code",
        "eval injection", "template injection",
    ],
    "authentication_bypass": [
        "authentication bypass", "improper authentication", "missing authentication",
        "default credentials", "weak password", "session fixation", "credential",
        "privilege escalation", "improper authorization", "access control",
    ],
    "cryptographic_weakness": [
        "weak encryption", "broken cryptography", "insecure random", "hardcoded key",
        "certificate validation", "tls", "ssl", "man in the middle", "cipher",
        "cryptographic", "key exchange",
    ],
    "network_exposure": [
        "denial of service", "dos", "remote denial", "network packet", "firewall bypass",
        "open port", "unauthenticated remote", "network exposure", "reachable",
        "amplification attack",
    ],
    "information_disclosure": [
        "information disclosure", "sensitive data", "path traversal", "directory traversal",
        "arbitrary file read", "source code disclosure", "stack trace", "error message",
        "debug information", "password disclosure",
    ],
    "supply_chain": [
        "dependency confusion", "malicious package", "typosquatting", "build system",
        "third party library", "open source component", "software supply chain",
        "package manager", "malicious update", "trojanized",
    ],
    "configuration_flaw": [
        "misconfiguration", "default configuration", "insecure default", "exposed service",
        "cross site scripting", "xss", "csrf", "cross site request forgery",
        "clickjacking", "open redirect",
    ],
}


class WikiTopicMapper:
    """Maps CVE descriptions to K threat topics using NMF or LDA on TF-IDF features."""

    def __init__(
        self,
        n_topics: int = 8,
        method: str = "nmf",
        max_features: int = 5000,
    ) -> None:
        if method not in ("nmf", "lda"):
            raise ValueError(f"method must be 'nmf' or 'lda', got {method!r}")
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features

        self.vectorizer: TfidfVectorizer | None = None
        self.model: NMF | LatentDirichletAllocation | None = None
        self.topic_word_matrix: np.ndarray | None = None  # (K, vocab)

    # ------------------------------------------------------------------
    def fit(self, descriptions: list[str], n_topics: int | None = None) -> "WikiTopicMapper":
        """Fit TF-IDF vectorizer then NMF or LDA on the corpus."""
        if n_topics is not None:
            self.n_topics = n_topics

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(descriptions)  # (n_docs, vocab)

        if self.method == "nmf":
            self.model = NMF(
                n_components=self.n_topics,
                init="nndsvda",
                random_state=42,
                max_iter=400,
                l1_ratio=0.1,
            )
            self.model.fit(X)
            self.topic_word_matrix = self.model.components_  # (K, vocab)
        else:  # lda
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
                learning_method="online",
                batch_size=256,
            )
            self.model.fit(X)
            self.topic_word_matrix = self.model.components_  # (K, vocab)

        return self

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("Call .fit() before transform/predict methods.")

    def transform(self, descriptions: list[str]) -> np.ndarray:
        """Return (n_docs, K) soft topic distribution."""
        self._check_fitted()
        X = self.vectorizer.transform(descriptions)  # type: ignore[union-attr]
        if self.method == "nmf":
            W = self.model.transform(X)  # type: ignore[union-attr]
            # Normalise rows to sum to 1 (NMF gives non-negative but unnormalised)
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return W / row_sums
        else:
            return self.model.transform(X)  # type: ignore[union-attr]  # LDA already normalised

    def assign_hard(self, descriptions: list[str]) -> np.ndarray:
        """Return argmax topic label per document, shape (n_docs,)."""
        probs = self.transform(descriptions)
        return np.argmax(probs, axis=1)

    def top_words(self, topic_k: int, n_words: int = 10) -> list[str]:
        """Return top n_words for topic topic_k."""
        self._check_fitted()
        feature_names: list[str] = self.vectorizer.get_feature_names_out().tolist()  # type: ignore[union-attr]
        topic_row = self.topic_word_matrix[topic_k]  # type: ignore[index]
        top_indices = np.argsort(topic_row)[::-1][:n_words]
        return [feature_names[i] for i in top_indices]

    def get_topic_labels(self) -> list[str]:
        """Return a human-readable label per topic based on its top 3 words."""
        self._check_fitted()
        labels: list[str] = []
        for k in range(self.n_topics):
            words = self.top_words(k, n_words=3)
            label = "_".join(w.replace(" ", "-") for w in words)
            labels.append(f"topic{k}:{label}")
        return labels
