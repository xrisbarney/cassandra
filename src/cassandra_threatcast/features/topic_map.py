"""
topic_map.py
============
Maps CVE descriptions to K threat topics using NMF or LDA on TF-IDF features.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.decomposition import NMF, LatentDirichletAllocation

# CVE descriptions share a lot of near-universal report-writing boilerplate
_CVE_BOILERPLATE_STOPWORDS = [
    "vulnerability", "vulnerabilities", "allows", "allow", "could", "may",
    "attacker", "attackers", "successful", "cvss", "score", "affected",
    "version", "versions", "prior", "due", "via", "using", "needed", "lead",
    "oracle", "exploitation", "result", "results",
]

# Seed vocabulary for 8 canonical threat topics
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
            stop_words=list(sk_text.ENGLISH_STOP_WORDS.union(_CVE_BOILERPLATE_STOPWORDS)),
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        X = self.vectorizer.fit_transform(descriptions)  # (n_docs, vocab)

        if self.method == "nmf":
            if self.n_topics >= 64:
                # Large-K (sparse-topic population) fits: full NMF at K in the
                from sklearn.decomposition import MiniBatchNMF
                self.model = MiniBatchNMF(
                    n_components=self.n_topics,
                    init="nndsvda",
                    random_state=42,
                    max_iter=200,
                    batch_size=2048,
                    l1_ratio=0.1,
                )
            else:
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

    def get_topic_labels(self, n_display: int = 3) -> list[str]:
        """Return a human-readable label per topic from its top words.

        Because top_words() returns both unigrams and bigrams, the naive top-N
        often repeats the same concept twice (e.g. 'needed', 'privileges
        needed', 'execution privileges' all share tokens) which reads as
        garbled rather than informative. This greedily picks the top
        *n_display* candidates whose word tokens don't overlap with any
        already-chosen candidate, so each slot in the label contributes a
        genuinely distinct word/concept.
        """
        self._check_fitted()
        labels: list[str] = []
        for k in range(self.n_topics):
            candidates = self.top_words(k, n_words=max(10, n_display * 3))
            selected: list[str] = []
            seen_tokens: set[str] = set()
            for word in candidates:
                tokens = set(word.replace("-", " ").split())
                if tokens & seen_tokens:
                    continue
                selected.append(word)
                seen_tokens |= tokens
                if len(selected) == n_display:
                    break
            label = " / ".join(w.replace("-", " ").title() for w in selected)
            labels.append(f"topic{k}:{label}")
        return labels


def load_topic_labels(data_dir: str, K: int) -> list[str]:
    """Load topic labels for display, preferring AI-generated names
    (``topic_names.json``, e.g. 'SQL Injection') over the raw top-words
    label (e.g. 'Sql / Injection / Php') for any topic that has one, and
    falling back to generic 'Topic N' names if nothing is available.

    Shared by scripts/forecast.py and app.py so both show the same names.
    """
    import json
    import os
    import pickle

    fallback = [f"Topic {k}" for k in range(K)]

    word_labels = fallback
    mapper_path = os.path.join(data_dir, "topic_mapper.pkl")
    if os.path.exists(mapper_path):
        try:
            with open(mapper_path, "rb") as fh:
                mapper = pickle.load(fh)
            word_labels = [lbl.split(":", 1)[-1] for lbl in mapper.get_topic_labels()]
        except Exception:  # noqa: BLE001 -- cosmetic only, never fatal
            pass

    ai_names: dict[str, str] = {}
    names_path = os.path.join(data_dir, "topic_names.json")
    if os.path.exists(names_path):
        try:
            with open(names_path, encoding="utf-8") as fh:
                ai_names = json.load(fh)
        except Exception:  # noqa: BLE001 -- cosmetic only, never fatal
            pass

    return [ai_names.get(str(k), word_labels[k] if k < len(word_labels) else f"Topic {k}") for k in range(K)]
