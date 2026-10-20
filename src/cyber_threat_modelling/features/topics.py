"""Initial-window-only text features with portable, non-pickle topic state."""

import warnings

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from ..data.store import write_json


def hard_assign(weights):
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Topic coefficients must be finite and nonnegative.")
    labels = np.argmax(weights, axis=1) + 1
    labels[weights.max(axis=1) == 0] = weights.shape[1] + 1
    return labels


class FrozenTopics:
    def __init__(self, settings):
        self.settings = settings
        options = dict(settings["vectorizer"])
        options["ngram_range"] = tuple(options["ngram_range"])
        self.vectorizer = TfidfVectorizer(**options)
        self.nmf = NMF(
            n_components=settings["learned_topics"],
            random_state=settings["seed"],
            **settings["nmf"],
        )

    def fit(self, records):
        train = records[
            records.publication_month.between(
                self.settings["training_start"], self.settings["training_end"]
            )
            & ~records.rejected
        ].sort_values("cve_id")
        if train.cve_id.duplicated().any() or train.empty:
            raise ValueError("Invalid initial topic training cohort.")
        with threadpool_limits(limits=self.settings["threads"]):
            matrix = self.vectorizer.fit_transform(train.description_en.fillna(""))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                self.nmf.fit(matrix)
        self.training_ids = train.cve_id.tolist()
        self.training_iterations = int(self.nmf.n_iter_)
        self.training_matrix = matrix
        return self

    def transform(self, descriptions):
        labels, confidence, reasons = [], [], []
        size = self.settings["transform_batch_size"]
        descriptions = list(descriptions)
        with threadpool_limits(limits=self.settings["threads"]):
            for start in range(0, len(descriptions), size):
                texts = [
                    t if isinstance(t, str) else "" for t in descriptions[start : start + size]
                ]
                matrix = self.vectorizer.transform(texts)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    weights = self.nmf.transform(matrix)
                assigned = hard_assign(weights)
                sums = weights.sum(axis=1)
                shares = np.divide(
                    weights.max(axis=1), sums, out=np.zeros(len(sums)), where=sums > 0
                )
                for i, text in enumerate(texts):
                    reasons.append(
                        "empty_text"
                        if not text.strip()
                        else "no_vocabulary_terms"
                        if matrix[i].nnz == 0
                        else "zero_topic_coefficients"
                        if sums[i] == 0
                        else "assigned"
                    )
                labels.extend(assigned)
                confidence.extend(shares)
        return np.asarray(labels), np.asarray(confidence), reasons

    def save(self, folder):
        write_json(
            folder / "topic_vocabulary.json",
            {term: int(index) for term, index in self.vectorizer.vocabulary_.items()},
        )
        np.savez_compressed(
            folder / "topic_arrays.npz", idf=self.vectorizer.idf_, components=self.nmf.components_
        )
        write_json(folder / "topic_training_ids.json", self.training_ids)

    @classmethod
    def load(cls, folder, settings):
        import json

        model = cls(settings)
        vocabulary = json.loads((folder / "topic_vocabulary.json").read_text(encoding="utf-8"))
        model.vectorizer.set_params(vocabulary=vocabulary)
        with np.load(folder / "topic_arrays.npz", allow_pickle=False) as state:
            model.vectorizer.idf_ = state["idf"]
            model.nmf.components_ = state["components"]
            model.nmf.n_features_in_ = len(state["idf"])
        return model

    def coherence(self):
        """Descriptive document NPMI on initial training only; never a selection score."""
        names = self.vectorizer.get_feature_names_out()
        binary = (self.training_matrix > 0).astype(np.float64)
        output = []
        for k, weights in enumerate(self.nmf.components_, 1):
            top = np.argsort(-weights, kind="stable")[:10]
            block = binary[:, top]
            joint = (block.T @ block).toarray() / binary.shape[0]
            marginal = joint.diagonal()
            values = []
            for i in range(10):
                for j in range(i):
                    p = joint[i, j]
                    values.append(
                        -1.0
                        if p == 0
                        else float(np.log(p / (marginal[i] * marginal[j])) / -np.log(p))
                        if p < 1
                        else 0.0
                    )
            output.append(
                {
                    "category": k,
                    "top_terms": names[top].tolist(),
                    "training_document_npmi": float(np.mean(values)),
                }
            )
        return output
