"""Extract brand mentions from probe answers and compute visibility scores."""
from __future__ import annotations
from dataclasses import dataclass, field
from geo_prober import ProbeResult
from logger import get_logger

log = get_logger("visibility_scorer")


@dataclass
class BrandVisibility:
    brand: str
    mention_count: int
    total_probes: int
    models_mentioned: list[str]
    visibility_pct: float  # 0–100


@dataclass
class VisibilityReport:
    category: str
    brands: list[BrandVisibility]
    probe_count: int


class VisibilityScorer:
    def score(self, category: str, brands: list[str], results: list[ProbeResult]) -> VisibilityReport:
        valid = [r for r in results if not r.error]
        total = len(valid)

        brand_data: dict[str, BrandVisibility] = {
            b: BrandVisibility(brand=b, mention_count=0, total_probes=total,
                               models_mentioned=[], visibility_pct=0.0)
            for b in brands
        }

        for result in valid:
            lower_answer = result.answer.lower()
            for brand in brands:
                if brand.lower() in lower_answer:
                    bd = brand_data[brand]
                    bd.mention_count += 1
                    if result.model not in bd.models_mentioned:
                        bd.models_mentioned.append(result.model)

        for bd in brand_data.values():
            bd.visibility_pct = round((bd.mention_count / total * 100) if total else 0, 1)
            log.info("Brand '%s': %d/%d probes (%.1f%%)", bd.brand, bd.mention_count, total, bd.visibility_pct)

        sorted_brands = sorted(brand_data.values(), key=lambda b: -b.visibility_pct)
        return VisibilityReport(category=category, brands=sorted_brands, probe_count=total)
