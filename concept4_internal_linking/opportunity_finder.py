from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from content_embedder import PageContent
from logger import get_logger

log = get_logger("opportunity_finder")


@dataclass
class LinkOpportunity:
    source: PageContent
    target: PageContent
    similarity: float
    priority: str   # HIGH | MEDIUM | LOW
    is_orphan_target: bool


class OpportunityFinder:
    def __init__(self, min_similarity: float = 0.75, weak_threshold: float = 3):
        self._min_sim = min_similarity
        self._weak_threshold = weak_threshold

    def find(self, pages: list[PageContent],
             outgoing: dict[str, set[str]],
             incoming: dict[str, set[str]]) -> list[LinkOpportunity]:

        pages_with_emb = [p for p in pages if p.embedding]
        if not pages_with_emb:
            return []

        matrix = np.array([p.embedding for p in pages_with_emb], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.where(norms == 0, 1, norms)
        sims = matrix @ matrix.T

        opportunities: list[LinkOpportunity] = []
        n = len(pages_with_emb)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                src = pages_with_emb[i]
                tgt = pages_with_emb[j]
                sim = float(sims[i, j])
                if sim < self._min_sim:
                    continue
                src_out = outgoing.get(src.url, set())
                if tgt.url in src_out:
                    continue
                is_orphan = len(incoming.get(tgt.url, set())) < self._weak_threshold
                score = sim + (0.15 if is_orphan else 0)
                priority = "HIGH" if score >= 0.9 else ("MEDIUM" if score >= 0.8 else "LOW")
                opportunities.append(LinkOpportunity(
                    source=src, target=tgt, similarity=sim,
                    priority=priority, is_orphan_target=is_orphan,
                ))

        opportunities.sort(key=lambda o: -o.similarity)
        log.info("Found %d link opportunities", len(opportunities))
        return opportunities
