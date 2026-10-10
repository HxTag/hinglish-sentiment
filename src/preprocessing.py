"""Shared configurable preprocessing for training and inference."""

from __future__ import annotations

import re
from typing import Any, Mapping

import emoji

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@[\w_]+")
HTML_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object, settings: Mapping[str, Any]) -> str:
    """Normalize text without removing or transliterating Hindi-origin words.

    This is shared by training and inference to prevent training-serving skew.
    """
    if text is None:
        return ""
    value = str(text)
    if not value or value.lower() == "nan":
        return ""
    if settings.get("remove_html", True):
        value = HTML_PATTERN.sub(" ", value)
    if settings.get("remove_urls", True):
        value = URL_PATTERN.sub(" ", value)
    if settings.get("remove_mentions", True):
        value = MENTION_PATTERN.sub(" ", value)
    if settings.get("emoji_to_text", True):
        # Make emoji sentiment explicit, e.g. 😊 becomes "smiling face".
        value = emoji.demojize(value, delimiters=(" ", " ")).replace("_", " ")
    if settings.get("lowercase", True):
        value = value.lower()
    if settings.get("normalize_repeated_characters", True):
        maximum = int(settings.get("max_repeated_characters", 2))
        if maximum < 1:
            raise ValueError("max_repeated_characters must be at least 1")
        value = re.sub(r"(.)\1{" + str(maximum) + r",}", r"\1" * maximum, value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()
