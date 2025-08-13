from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st

from backend.auth import load_students, verify_credentials, verify_id_card_with_ocr, Student
from backend.search_index import SemanticSearchIndex
from backend.pdf_utils import extract_references_from_pdf


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
BOOKS_DIR = DATA_DIR / "books"
PAPERS_DIR = DATA_DIR / "papers"
STORAGE_DIR = APP_DIR / "storage"


st.set_page_config(page_title="E‑Library", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def get_index() -> SemanticSearchIndex:
    index = SemanticSearchIndex(storage_dir=STORAGE_DIR)
    index.load_or_build(BOOKS_DIR, PAPERS_DIR)
    return index


@st.cache_data(show_spinner=False)
def get_students():
    return load_students()


def reset_auth():
    for key in ["auth_user", "logged_in", "selected_pdf_path", "search_results"]:
        if key in st.session_state:
            del st.session_state[key]


def b64_pdf_html(pdf_path: Path, height: int = 800) -> str:
    try:
        with open(pdf_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>'
        return html
    except Exception as exc:
        return f"<div>Failed to open PDF: {exc}</div>"


def sidebar_nav() -> str:
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/brand/refs/heads/main/logo/mark/streamlit-mark-color.png", width=36)
        st.title("E‑Library")
        if st.session_state.get("auth_user"):
            user: Student = st.session_state["auth_user"]
            st.caption(f"Signed in as {user.name} ({user.stream})")
        choice = st.radio("Navigation", ["Home", "Search Library", "References", "Previous Year Papers", "Upload PDFs", "Rebuild Index", "Logout"], index=0)
        return choice


def login_view():
    st.header("Login & Student Verification")
    col1, col2 = st.columns([2, 1])
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        id_upload = st.file_uploader("Upload Student ID card (image or PDF)", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=False)
        if st.button("Login"):
            students = get_students()
            user = verify_credentials(username, password, students)
            if not user:
                st.error("Invalid username or password.")
                return
            if not id_upload:
                st.error("Please upload your Student ID card for verification.")
                return
            with st.spinner("Verifying ID via OCR..."):
                ok, msg = verify_id_card_with_ocr(user, id_upload)
            if not ok:
                st.error(f"Verification failed: {msg}")
                return
            st.success("Verification successful! Redirecting...")
            st.session_state["auth_user"] = user
            st.session_state["logged_in"] = True
            st.rerun()
    with col2:
        st.info("We use OCR to verify your name and Student ID from the uploaded card.")


def suggest_books_for_user(user: Student) -> List[Path]:
    suggestions: List[Path] = []
    if not BOOKS_DIR.exists():
        return suggestions
    interest_keywords = [kw.lower() for kw in user.interests] + [user.stream.lower()]
    for root, _, files in os.walk(BOOKS_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                lower_name = f.lower()
                if any(kw in lower_name for kw in interest_keywords):
                    suggestions.append(Path(root) / f)
    return suggestions[:10]


def home_view():
    user: Student = st.session_state["auth_user"]
    st.subheader(f"Welcome, {user.name} 👋")
    st.write("Here are some suggested books based on your stream and interests:")
    suggestions = suggest_books_for_user(user)
    if not suggestions:
        st.caption("No suggestions yet. Add PDFs to app/data/books.")
    for pdf_path in suggestions:
        with st.container(border=True):
            st.write(f"📘 {pdf_path.name}")
            col_a, col_b = st.columns([1, 5])
            with col_a:
                if st.button("Open", key=f"open_{pdf_path}"):
                    st.session_state["selected_pdf_path"] = str(pdf_path)
            with col_b:
                st.caption(str(pdf_path.relative_to(APP_DIR)))

    if st.session_state.get("selected_pdf_path"):
        st.markdown("---")
        st.write("Selected PDF preview:")
        html = b64_pdf_html(Path(st.session_state["selected_pdf_path"]))
        st.components.v1.html(html, height=800, scrolling=True)


def search_view():
    index = get_index()
    st.subheader("Search Library 🔍")
    query = st.text_input("Search for any topic, title, or concept")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        kind = st.selectbox("Scope", ["All", "Books", "Papers"])
    with col2:
        subjects = [d.name for d in PAPERS_DIR.iterdir() if d.is_dir()] if PAPERS_DIR.exists() else []
        subj_sel = st.multiselect("Subjects (papers)", subjects)
    with col3:
        k = st.slider("Results", min_value=5, max_value=30, value=10, step=5)

    if st.button("Search") and query.strip():
        kind_filter = None if kind == "All" else ("book" if kind == "Books" else "paper")
        with st.spinner("Searching..."):
            results = index.query(query, k=k, kind_filter=kind_filter, subject_filter=subj_sel or None)
        st.session_state["search_results"] = results

    results = st.session_state.get("search_results", [])
    for i, r in enumerate(results):
        with st.container(border=True):
            st.write(f"{i+1}. [{r.kind.upper()}] {Path(r.file_path).name} — page {r.page_number} — score {r.score:.3f}")
            st.caption(str(Path(r.file_path).relative_to(APP_DIR)))
            st.code((r.text[:800] + ("..." if len(r.text) > 800 else "")))
            if st.button("Open PDF", key=f"open_search_{i}"):
                st.session_state["selected_pdf_path"] = r.file_path

    if st.session_state.get("selected_pdf_path"):
        st.markdown("---")
        st.write("Selected PDF preview:")
        html = b64_pdf_html(Path(st.session_state["selected_pdf_path"]))
        st.components.v1.html(html, height=800, scrolling=True)


def references_view():
    st.subheader("References 📄")
    if not BOOKS_DIR.exists():
        st.info("Add PDFs to app/data/books to extract references.")
        return
    for root, _, files in os.walk(BOOKS_DIR):
        for f in files:
            if not f.lower().endswith(".pdf"):
                continue
            pdf_path = Path(root) / f
            refs = extract_references_from_pdf(pdf_path)
            if refs:
                with st.expander(f"{f} — {len(refs)} references"):
                    for r in refs:
                        st.markdown(f"- {r}")
            else:
                with st.expander(f"{f} — no references found"):
                    st.caption(str(pdf_path.relative_to(APP_DIR)))


def papers_view():
    st.subheader("Previous Year Papers 📝")
    subjects = [d.name for d in PAPERS_DIR.iterdir() if d.is_dir()] if PAPERS_DIR.exists() else []
    if not subjects:
        st.info("Create subject folders under app/data/papers and add PDFs with year in filename.")
        return
    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("Subject", subjects)
    with col2:
        # Extract years found in filenames
        subject_dir = PAPERS_DIR / subject
        year_options = []
        for f in sorted([p.name for p in subject_dir.glob("*.pdf")]):
            m = re.search(r"(19|20)\d{2}", f)
            if m:
                year_options.append(m.group(0))
        year_options = sorted(list(set(year_options)))
        year = st.selectbox("Year", ["All"] + year_options)

    files: List[Path] = []
    for p in sorted(subject_dir.glob("*.pdf")):
        if year != "All":
            m = re.search(r"(19|20)\d{2}", p.name)
            if not (m and m.group(0) == year):
                continue
        files.append(p)

    for pdf in files:
        with st.container(border=True):
            st.write(f"📄 {pdf.name}")
            st.caption(str(pdf.relative_to(APP_DIR)))
            with open(pdf, "rb") as f:
                st.download_button("Download", f, file_name=pdf.name, mime="application/pdf")
            if st.button("Open", key=f"open_paper_{pdf}"):
                st.session_state["selected_pdf_path"] = str(pdf)

    st.markdown("---")
    st.write("Search within selected subject:")
    query = st.text_input("Query", key="paper_query")
    if st.button("Search Papers") and query.strip():
        index = get_index()
        with st.spinner("Searching papers..."):
            results = index.query(query, k=10, kind_filter="paper", subject_filter=[subject])
        st.session_state["paper_search_results"] = results

    for r in st.session_state.get("paper_search_results", []):
        with st.container(border=True):
            st.write(f"[{r.kind.upper()}] {Path(r.file_path).name} — page {r.page_number} — score {r.score:.3f}")
            st.code((r.text[:800] + ("..." if len(r.text) > 800 else "")))
            if st.button("Open PDF", key=f"open_paper_res_{r.file_path}_{r.page_number}"):
                st.session_state["selected_pdf_path"] = r.file_path

    if st.session_state.get("selected_pdf_path"):
        st.markdown("---")
        html = b64_pdf_html(Path(st.session_state["selected_pdf_path"]))
        st.components.v1.html(html, height=800, scrolling=True)


def upload_view():
    st.subheader("Upload PDFs")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Books")
        book_files = st.file_uploader("Upload book PDFs", type=["pdf"], accept_multiple_files=True, key="upload_books")
        if book_files:
            BOOKS_DIR.mkdir(parents=True, exist_ok=True)
            for uf in book_files:
                dest = BOOKS_DIR / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.read())
            st.success(f"Uploaded {len(book_files)} book(s).")

    with col2:
        st.write("Papers")
        subject = st.text_input("Subject (e.g., Mathematics)")
        paper_files = st.file_uploader("Upload papers PDFs", type=["pdf"], accept_multiple_files=True, key="upload_papers")
        if subject and paper_files:
            subj_dir = PAPERS_DIR / subject
            subj_dir.mkdir(parents=True, exist_ok=True)
            for uf in paper_files:
                dest = subj_dir / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.read())
            st.success(f"Uploaded {len(paper_files)} paper(s) to {subject}.")

    if st.button("Rebuild Search Index"):
        with st.spinner("Rebuilding index..."):
            idx = get_index()
            idx.load_or_build(BOOKS_DIR, PAPERS_DIR, force_rebuild=True)
        st.success("Index rebuilt.")


def rebuild_index_view():
    st.subheader("Rebuild Search Index")
    st.write("This will parse all PDFs and rebuild the FAISS index.")
    if st.button("Rebuild Now"):
        with st.spinner("Rebuilding index..."):
            idx = get_index()
            idx.load_or_build(BOOKS_DIR, PAPERS_DIR, force_rebuild=True)
        st.success("Index rebuilt.")


def main():
    # Ensure directories exist
    for p in [BOOKS_DIR, PAPERS_DIR, STORAGE_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    if not st.session_state.get("logged_in"):
        login_view()
        return

    choice = sidebar_nav()
    if choice == "Home":
        home_view()
    elif choice == "Search Library":
        search_view()
    elif choice == "References":
        references_view()
    elif choice == "Previous Year Papers":
        papers_view()
    elif choice == "Upload PDFs":
        upload_view()
    elif choice == "Rebuild Index":
        rebuild_index_view()
    elif choice == "Logout":
        reset_auth()
        st.rerun()


if __name__ == "__main__":
    main()