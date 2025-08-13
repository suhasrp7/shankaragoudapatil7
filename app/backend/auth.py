from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytesseract
from pytesseract import TesseractNotFoundError

from .pdf_utils import pdf_first_page_to_pil_image, image_file_to_pil


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STUDENTS_JSON = DATA_DIR / "students.json"


@dataclass
class Student:
    username: str
    password: str
    student_id: str
    name: str
    stream: str
    interests: list[str]


def load_students() -> Dict[str, Student]:
    with open(STUDENTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    index: Dict[str, Student] = {}
    for s in data:
        index[s["username"].lower()] = Student(
            username=s["username"],
            password=s["password"],
            student_id=str(s["student_id"]),
            name=s["name"],
            stream=s.get("stream", ""),
            interests=s.get("interests", []),
        )
    return index


def verify_credentials(username: str, password: str, students: Dict[str, Student]) -> Optional[Student]:
    user = students.get(username.lower().strip())
    if not user:
        return None
    if user.password != password:
        return None
    return user


def ocr_text_from_upload(uploaded_file) -> Tuple[Optional[str], Optional[str]]:
    # Returns (text, error)
    try:
        file_name = uploaded_file.name.lower()
        data = uploaded_file.read()
        if file_name.endswith(".pdf"):
            # Write temp file
            tmp_path = Path(DATA_DIR) / "_tmp_idcard.pdf"
            with open(tmp_path, "wb") as f:
                f.write(data)
            image = pdf_first_page_to_pil_image(tmp_path)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            if image is None:
                return None, "Could not render PDF to image for OCR."
        else:
            image = image_file_to_pil(data)
            if image is None:
                return None, "Unsupported image format."

        text = pytesseract.image_to_string(image)
        return text, None
    except TesseractNotFoundError:
        return None, "Tesseract OCR is not installed. Please install `tesseract-ocr`."
    except Exception as exc:
        return None, f"OCR failed: {exc}"


def verify_id_card_with_ocr(student: Student, uploaded_file) -> Tuple[bool, str]:
    text, error = ocr_text_from_upload(uploaded_file)
    if error:
        return False, error
    if not text:
        return False, "No text detected from ID card."

    text_lower = text.lower()
    name_ok = student.name.lower() in text_lower

    # Try to match alphanumeric IDs like S123456 or 20201234 etc.
    id_pattern = re.escape(student.student_id.lower())
    id_ok = re.search(id_pattern, text_lower) is not None

    if name_ok and id_ok:
        return True, "Verified"

    # Try relaxed name check (first and last independently)
    name_parts = [p for p in re.split(r"\s+", student.name.lower()) if p]
    relaxed_name_ok = all(any(p in token for token in text_lower.split()) for p in name_parts)

    if relaxed_name_ok and id_ok:
        return True, "Verified (relaxed name match)"

    return False, "Name/ID not found on the ID card via OCR."