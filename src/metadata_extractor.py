import re
from pathlib import Path
from typing import Dict, List, Any


# Comprehensive technical skill taxonomy across software engineering, data science, devops, testing, database, web dev, etc.
COMMON_SKILLS = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "r", "go", "golang", "ruby", "php", "scala", "swift", "kotlin", "bash", "shell",
    # Web & Frontend Frameworks
    "react", "react.js", "reactjs", "angular", "vue", "vue.js", "next.js", "node.js", "express", "django", "flask", "fastapi", "spring", "spring boot",
    "html", "html5", "css", "css3", "bootstrap", "tailwind", "redux", "jquery", "asp.net", "laravel",
    # Databases & Big Data
    "postgresql", "postgres", "mysql", "mongodb", "oracle", "sqlite", "redis", "elasticsearch", "cassandra", "snowflake", "bigquery", "spark", "hadoop", "hive",
    # Machine Learning & AI / Data Science
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision", "tensorflow", "pytorch", "scikit-learn", "sklearn",
    "keras", "pandas", "numpy", "scipy", "tableau", "power bi", "data analysis", "data mining", "statsmodels",
    # Cloud & DevOps & Tools
    "aws", "amazon web services", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "git", "github",
    "gitlab", "ci/cd", "linux", "unix", "jira", "confluence", "prometheus", "grafana", "bamboo",
    # Testing & QA
    "selenium", "junit", "testng", "cucumber", "cypress", "postman", "jmeter", "pytest", "manual testing", "automation testing", "qa",
    # SAP & ERP & Business Analysis
    "sap", "sap hana", "sap abap", "business analysis", "agile", "scrum", "sdlc", "requirements gathering", "etl"
]


class MetadataExtractor:
    """
    Extracts structured metadata from resume text:
    - candidate_name
    - skills_list
    - total_years_experience
    - highest_education
    """

    def __init__(self):
        # Build regex patterns for skills taxonomy
        self.skill_patterns = {
            skill: re.compile(rf'\b{re.escape(skill)}\b', re.IGNORECASE)
            for skill in COMMON_SKILLS
        }

    def extract_metadata(self, text: str, filename: str = "") -> Dict[str, Any]:
        """
        Extract structured metadata dictionary for a given resume.
        """
        candidate_name = self.extract_name(text, filename)
        skills = self.extract_skills(text)
        years_exp = self.extract_years_experience(text)
        education = self.extract_education(text)

        return {
            "candidate_name": candidate_name,
            "skills": skills,
            "total_years_experience": years_exp,
            "highest_education": education
        }

    def extract_name(self, text: str, filename: str = "") -> str:
        """
        Derive candidate name from filename or top lines of resume text.
        """
        # Try from filename first if clean (e.g. John_Doe_Resume.pdf)
        if filename:
            clean_stem = Path(filename).stem
            # Remove numeric IDs like 103.pdf or resume_103
            clean_stem = re.sub(r'^\d+[\s_-]*|[\s_-]*\d+$', '', clean_stem).strip()
            clean_stem = clean_stem.replace('_', ' ').replace('-', ' ').strip()

            if len(clean_stem) > 2 and not clean_stem.isdigit() and not clean_stem.lower() in ["resume", "cv"]:
                return clean_stem.title()

        # Fallback: check first 3 lines of text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:3]:
            # Remove symbols/emails/phones
            cleaned = re.sub(r'\(.*?\)|[0-9+@:\-|,]', '', line).strip()
            words = cleaned.split()
            if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                # Ensure it's not a resume heading like "Curriculum Vitae"
                if cleaned.lower() not in ["curriculum vitae", "resume", "work experience", "summary"]:
                    return cleaned.title()

        # Final fallback
        if filename:
            return Path(filename).stem
        return "Candidate"

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract recognized technical skills from resume text.
        """
        found_skills = set()
        for skill, pattern in self.skill_patterns.items():
            if pattern.search(text):
                # Standardize skill representation
                skill_name = skill.upper() if len(skill) <= 4 and skill not in ["java", "html", "bash", "unix", "ruby"] else skill.title()
                if skill == "react.js" or skill == "reactjs":
                    skill_name = "React"
                elif skill == "node.js":
                    skill_name = "Node.js"
                elif skill == "vue.js":
                    skill_name = "Vue.js"
                elif skill == "aws" or skill == "amazon web services":
                    skill_name = "AWS"
                elif skill == "gcp" or skill == "google cloud":
                    skill_name = "GCP"
                elif skill == "sql":
                    skill_name = "SQL"
                elif skill == "nlp":
                    skill_name = "NLP"
                elif skill == "qa":
                    skill_name = "QA"
                elif skill == "etl":
                    skill_name = "ETL"
                elif skill == "sap":
                    skill_name = "SAP"
                found_skills.add(skill_name)
        return sorted(list(found_skills))

    def extract_years_experience(self, text: str) -> float:
        """
        Estimate total years of experience using explicit text mentions and date-range extraction.
        """
        # 1. Look for explicit mentions like "5+ years of experience", "3 years experience"
        exp_pattern = re.compile(
            r'(\b\d{1,2}(?:\.\d)?\b)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|work)',
            re.IGNORECASE
        )
        matches = exp_pattern.findall(text)
        if matches:
            years = [float(m) for m in matches if float(m) < 40]
            if years:
                return max(years)

        # 2. Look for year ranges like "2018 - 2023", "2019 to Present", "2016-2021"
        range_pattern = re.compile(
            r'\b(20[0-2]\d|19[89]\d)\s*(?:-|–|to)\s*(20[0-2]\d|19[89]\d|present|current|till date)\b',
            re.IGNORECASE
        )
        ranges = range_pattern.findall(text)
        total_years = 0.0

        current_year = 2026
        for start, end in ranges:
            try:
                start_yr = int(start)
                end_yr = current_year if end.lower() in ["present", "current", "till date"] else int(end)
                diff = end_yr - start_yr
                if 0 <= diff <= 35:
                    total_years += diff
            except ValueError:
                pass

        if total_years > 0:
            return round(min(total_years, 30.0), 1)

        return 0.0

    def extract_education(self, text: str) -> str:
        """
        Identify highest education level from resume text.
        """
        text_lower = text.lower()

        if re.search(r'\b(ph\.?d|doctorate|doctor of philosophy)\b', text_lower):
            return "Ph.D."
        if re.search(r'\b(master|m\.?s|m\.?tech|m\.?c\.?a|m\.?b\.?a|post graduate)\b', text_lower):
            return "Master's"
        if re.search(r'\b(bachelor|b\.?s|b\.?tech|b\.?e|b\.?c\.?a|undergraduate)\b', text_lower):
            return "Bachelor's"
        if re.search(r'\b(associate|diploma)\b', text_lower):
            return "Associate / Diploma"

        return "Bachelor's"  # Default fallback if unspecified
