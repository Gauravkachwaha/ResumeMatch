import re
from typing import List, Dict, Tuple


SECTION_HEADERS = {
    "SUMMARY": [r'summary', r'professional summary', r'profile', r'objective', r'about me', r'career summary'],
    "EXPERIENCE": [r'experience', r'work experience', r'employment history', r'professional experience', r'work history', r'career history'],
    "EDUCATION": [r'education', r'academic background', r'academic qualifications', r'education history'],
    "SKILLS": [r'skills', r'technical skills', r'core competencies', r'technologies', r'key skills', r'skills & abilities'],
    "PROJECTS": [r'projects', r'academic projects', r'key projects', r'personal projects', r'significant projects'],
    "CERTIFICATIONS": [r'certifications', r'certificates', r'licenses', r'courses', r'training', r'achievements']
}


class TextProcessor:
    """
    Cleans raw text and splits resumes into section-aware chunks.
    """

    def __init__(self):
        # Compile regexes for section header detection
        header_patterns = []
        for section, keywords in SECTION_HEADERS.items():
            pattern_str = r'^(?:' + '|'.join(keywords) + r')[\s:]*$'
            header_patterns.append((section, re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)))
        self.header_patterns = header_patterns

    def clean_text(self, text: str) -> str:
        """
        Clean raw resume text from extra spaces, control characters, and markdown links.
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove null bytes / weird control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        # Fix spacing around punctuation
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)

        # Strip markdown links e.g. [email](mailto:...)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

        return text.strip()

    def detect_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Split text into logical sections based on section header detection.
        Returns list of (section_name, section_text) tuples.
        """
        lines = text.split('\n')
        sections = []

        current_section = "GENERAL"
        current_lines = []

        for line in lines:
            trimmed = line.strip()
            # Check if line matches a known section header
            matched_section = None
            if 2 <= len(trimmed) <= 40:
                for section_name, pattern in self.header_patterns:
                    if pattern.match(trimmed):
                        matched_section = section_name
                        break

            if matched_section:
                # Flush previous section
                if current_lines:
                    sec_content = "\n".join(current_lines).strip()
                    if sec_content:
                        sections.append((current_section, sec_content))
                current_section = matched_section
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sec_content = "\n".join(current_lines).strip()
            if sec_content:
                sections.append((current_section, sec_content))

        return sections

    def chunk_text(
        self,
        text: str,
        max_words: int = 250,
        overlap: int = 50
    ) -> List[Dict[str, str]]:
        """
        Split text into section-aware chunks.
        Each chunk is returned as a dict: {"section": section_name, "text": chunk_text}.
        """
        sections = self.detect_sections(text)
        chunks = []

        for section_name, section_text in sections:
            words = section_text.split()
            if not words:
                continue

            if len(words) <= max_words:
                chunk_content = f"[Section: {section_name}]\n{section_text}"
                chunks.append({
                    "section": section_name,
                    "text": chunk_content
                })
            else:
                # Sub-chunk long sections
                start = 0
                while start < len(words):
                    end = start + max_words
                    sub_words = words[start:end]
                    sub_text = " ".join(sub_words)
                    chunk_content = f"[Section: {section_name}]\n{sub_text}"
                    chunks.append({
                        "section": section_name,
                        "text": chunk_content
                    })
                    start += max_words - overlap

        return chunks