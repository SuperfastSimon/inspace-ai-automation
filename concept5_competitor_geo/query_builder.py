"""Build category-level GEO probe queries."""
from __future__ import annotations


def build_queries(category: str, use_cases: list[str]) -> list[str]:
    queries = [
        f"What are the best companies for {category}? List your top recommendations.",
        f"If I need help with {category}, which providers would you suggest and why?",
        f"Who are the market leaders in {category} right now?",
        f"Compare the top 5 {category} providers for a small business.",
        f"Which {category} companies are most trusted and recommended?",
    ]
    for uc in use_cases[:3]:
        queries.append(f"What tool or agency would you recommend for {uc}?")
    return queries
