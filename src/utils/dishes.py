"""Canonical Vietnamese dish set for FoodLensVN.

This is the **single source of truth** for the ``dish`` field used in dataset
annotations, splits stratification, and model evaluation breakdowns. All raw
annotation rows must use a key from :data:`CANONICAL_DISHES`; ``build_dataset``
rejects unknown keys.

Adding or removing a dish is a course-scope decision: update this list, then
re-run ``scripts/build_dataset.py`` to re-validate the corpus.
"""

from __future__ import annotations


CANONICAL_DISHES: tuple[str, ...] = (
    "pho",
    "bun_bo_hue",
    "banh_mi",
    "com_tam",
    "bun_cha",
    "goi_cuon",
    "cha_gio",
    "banh_xeo",
    "mi_quang",
    "hu_tieu",
)

CANONICAL_DISHES_SET: frozenset[str] = frozenset(CANONICAL_DISHES)

DISH_DISPLAY_NAMES: dict[str, str] = {
    "pho": "Phở",
    "bun_bo_hue": "Bún bò Huế",
    "banh_mi": "Bánh mì",
    "com_tam": "Cơm tấm",
    "bun_cha": "Bún chả",
    "goi_cuon": "Gỏi cuốn",
    "cha_gio": "Chả giò",
    "banh_xeo": "Bánh xèo",
    "mi_quang": "Mì Quảng",
    "hu_tieu": "Hủ tiếu",
}


def is_canonical(dish: str) -> bool:
    """Return True iff ``dish`` is one of the locked canonical keys."""
    return dish in CANONICAL_DISHES_SET
