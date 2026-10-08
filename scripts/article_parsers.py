from __future__ import annotations

from html.parser import HTMLParser
from typing import Any


class FixtureArticleParser(HTMLParser):
    parser_id = "fixture_html_v1"

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_article = False
        self.capture: str | None = None
        self.title_parts: list[str] = []
        self.body_parts: list[str] = []
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "article":
            self.in_article = True
        if tag == "meta" and attributes.get("name") and attributes.get("content"):
            self.meta[attributes["name"]] = attributes["content"]
        if self.in_article and tag == "h1":
            self.capture = "title"
        if self.in_article and tag == "p":
            self.capture = "body"

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "p"}:
            self.capture = None
        if tag == "article":
            self.in_article = False

    def handle_data(self, data: str) -> None:
        if self.capture == "title":
            self.title_parts.append(data)
        if self.capture == "body":
            self.body_parts.append(data)

    def result(self) -> dict[str, Any]:
        title = " ".join(part.strip() for part in self.title_parts if part.strip())
        body = "\n\n".join(part.strip() for part in self.body_parts if part.strip())
        if not title or not body:
            raise ValueError("Article fixture lacks an h1 or body paragraphs")
        authors = [item.strip() for item in self.meta.get("author", "").split(",") if item.strip()]
        return {
            "title": title,
            "body": body,
            "authors": authors,
            "published_at": self.meta.get("published_at"),
            "section": self.meta.get("section"),
        }


class DiprArticleParser(HTMLParser):
    parser_id = "dipr_html_v1"

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.body_depth = 0
        self.capture: str | None = None
        self.title_parts: list[str] = []
        self.body_fragments: list[str] = []
        self.current_paragraph: list[str] = []
        self.meta: dict[str, str] = {}

    @staticmethod
    def classes(attrs: dict[str, str | None]) -> set[str]:
        return set((attrs.get("class") or "").split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = self.classes(attributes)
        if tag == "meta":
            key = attributes.get("property") or attributes.get("name")
            if key and attributes.get("content"):
                self.meta[key] = attributes["content"]
        if tag == "article" or "dipr-post" in classes:
            self.article_depth += 1
        if self.article_depth and ("post-body" in classes or "article-body" in classes):
            self.body_depth += 1
        if self.article_depth and tag == "h1":
            self.capture = "title"
        if self.body_depth and tag in {"p", "li"}:
            self.capture = "body"
            self.current_paragraph = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1" and self.capture == "title":
            self.capture = None
        if tag in {"p", "li"} and self.capture == "body":
            paragraph = " ".join(part.strip() for part in self.current_paragraph if part.strip())
            if paragraph:
                self.body_fragments.append(paragraph)
            self.current_paragraph = []
            self.capture = None
        if tag == "article" and self.article_depth:
            self.article_depth -= 1
            self.body_depth = 0
        if tag == "div" and self.body_depth:
            self.body_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.capture == "title":
            self.title_parts.append(data)
        if self.capture == "body":
            self.current_paragraph.append(data)

    def result(self) -> dict[str, Any]:
        title = " ".join(part.strip() for part in self.title_parts if part.strip())
        title = title or self.meta.get("og:title", "").strip()
        body = "\n\n".join(self.body_fragments)
        if not title or not body:
            raise ValueError("DIPR article lacks a title or article body")
        author = self.meta.get("author", "").strip()
        return {
            "title": title,
            "body": body,
            "authors": [author] if author else [],
            "published_at": self.meta.get("article:published_time"),
            "updated_at": self.meta.get("article:modified_time"),
            "section": self.meta.get("article:section"),
        }


PARSERS = {
    FixtureArticleParser.parser_id: FixtureArticleParser,
    DiprArticleParser.parser_id: DiprArticleParser,
}


def parse_article(parser_id: str, html: str) -> dict[str, Any]:
    parser_class = PARSERS.get(parser_id)
    if parser_class is None:
        raise ValueError(f"Unknown parser {parser_id}")
    parser = parser_class()
    parser.feed(html)
    parser.close()
    return parser.result()
