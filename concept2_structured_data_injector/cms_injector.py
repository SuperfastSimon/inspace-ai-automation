from __future__ import annotations
import json
import os
import re
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from logger import get_logger

log = get_logger("cms_injector")
MARKER_START = "<!-- AUTO-INJECTED-SCHEMA -->"
MARKER_END = "<!-- /AUTO-INJECTED-SCHEMA -->"


class WordPressInjector:
    def __init__(self):
        self._base = os.environ["WP_SITE_URL"].rstrip("/")
        self._auth = (os.environ["WP_USERNAME"], os.environ["WP_APP_PASSWORD"])

    def inject(self, url: str, schema: dict) -> bool:
        post = self._find_post(url)
        if not post:
            log.warning("Could not find WP post for URL: %s", url)
            return False
        post_id = post["id"]
        post_type = post.get("type", "posts")
        current_content = post.get("content", {}).get("raw", "")
        new_content = self._inject_schema_block(current_content, schema)
        return self._update_post(post_id, post_type, new_content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def _find_post(self, url: str) -> dict | None:
        slug = url.rstrip("/").split("/")[-1]
        for endpoint in ["posts", "pages"]:
            resp = requests.get(
                f"{self._base}/wp-json/wp/v2/{endpoint}",
                params={"slug": slug, "context": "edit"},
                auth=self._auth, timeout=15,
            )
            if resp.ok and resp.json():
                post = resp.json()[0]
                post["type"] = endpoint
                return post
        return None

    def _inject_schema_block(self, content: str, schema: dict) -> str:
        block = f'{MARKER_START}<script type="application/ld+json">{json.dumps(schema, indent=2)}</script>{MARKER_END}'
        pattern = re.compile(re.escape(MARKER_START) + ".*?" + re.escape(MARKER_END), re.DOTALL)
        if pattern.search(content):
            return pattern.sub(block, content)
        return block + "\n" + content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def _update_post(self, post_id: int, post_type: str, content: str) -> bool:
        resp = requests.post(
            f"{self._base}/wp-json/wp/v2/{post_type}/{post_id}",
            json={"content": content},
            auth=self._auth, timeout=15,
        )
        if resp.ok:
            log.info("Injected schema into post #%d", post_id)
            return True
        log.error("Failed to update post #%d: %s", post_id, resp.text[:200])
        return False
