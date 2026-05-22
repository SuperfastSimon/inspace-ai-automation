"""Build brand-probe prompts covering awareness, recommendation, and comparison angles."""
from __future__ import annotations


def build_prompts(brand: str, domain: str, competitors: list[str]) -> list[str]:
    comp_str = ", ".join(competitors) if competitors else "competitors in your space"
    return [
        f"What do you know about {brand} ({domain})? Give a short overview.",
        f"If someone asked you to recommend a provider for {domain} services, "
        f"would you mention {brand}? Why or why not?",
        f"Compare {brand} with {comp_str} in one paragraph. Be objective.",
        f"List the top 5 companies you would recommend for {domain}. "
        f"Is {brand} on that list?",
        f"What are the strengths and weaknesses of {brand} as a {domain} provider?",
    ]
