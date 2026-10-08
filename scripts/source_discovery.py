from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit


class DiprListingParser(HTMLParser):
    void_elements = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.in_listing = 0
        self.urls: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if self.in_listing and tag not in self.void_elements:
            self.in_listing += 1
        elif "press-release-list" in classes:
            self.in_listing = 1
        if self.in_listing and tag == "a" and attributes.get("href"):
            absolute = urljoin(self.base_url, attributes["href"])
            parts = urlsplit(absolute)
            if parts.path.startswith("/post/"):
                self.urls.add(urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")))

    def handle_endtag(self, tag: str) -> None:
        if self.in_listing and tag not in self.void_elements:
            self.in_listing -= 1

    def result(self) -> list[str]:
        return sorted(self.urls)


def parse_dipr_listing(html: str, base_url: str = "https://dipr.mizoram.gov.in/") -> list[str]:
    parser = DiprListingParser(base_url)
    parser.feed(html)
    parser.close()
    return parser.result()
