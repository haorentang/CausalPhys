"""
Shared data types for SFT training scripts.
"""

from dataclasses import dataclass
from typing import List, Optional
from PIL import Image


@dataclass
class SFTItem:
    images: List[Image.Image]
    question: str
    answer_letter: str
    rationale: str  # raw text inside <rationale> ... </rationale>
    category: str
    subcategory: str
    index: int  # unique index for consistent sample selection
    ann_path: Optional[str] = None


@dataclass
class SFTItemAnswerOnly:
    images: List[Image.Image]
    question: str
    answer_letter: str
    category: str
    subcategory: str
    index: int  # unique index for consistent sample selection
    ann_path: Optional[str] = None
