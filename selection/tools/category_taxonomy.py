"""Canonical category mapping for BABEL action categories."""

import re
import unicodedata


def normalize_label(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


class CategoryTaxonomy:
    def __init__(self, config):
        self.unmapped = config.get("unmapped_category", "other")
        self.excluded = {
            normalize_label(value)
            for value in config.get("excluded_meta_categories", [])
        }
        self.lookup = {}
        for canonical, labels in config.get("groups", {}).items():
            self.lookup[normalize_label(canonical)] = canonical
            for label in labels:
                normalized = normalize_label(label)
                previous = self.lookup.get(normalized)
                if previous is not None and previous != canonical:
                    raise ValueError(
                        "taxonomy label {!r} belongs to both {} and {}".format(
                            label, previous, canonical
                        )
                    )
                self.lookup[normalized] = canonical

    def map_labels(self, labels):
        mapped = set()
        for label in labels:
            normalized = normalize_label(label)
            if not normalized:
                continue
            canonical = self.lookup.get(normalized, self.unmapped)
            if normalize_label(canonical) not in self.excluded:
                mapped.add(canonical)
        return mapped
