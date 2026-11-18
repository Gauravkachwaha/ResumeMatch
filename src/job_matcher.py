import json
import re
from typing import List, Dict, Any, Union, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.vector_store import ResumeVectorStore
from src.metadata_extractor import MetadataExtractor


class JobMatcher:
    """
    Part B — Job Matching Engine
    Executes hybrid search (semantic similarity + BM25 keyword matching), candidate scoring (0-100),
    must-have requirement hard filtering, match reasoning generation, and strict JSON output formatting.
    """

    def __init__(self, vector_store: Optional[ResumeVectorStore] = None):
        self.vector_store = vector_store or ResumeVectorStore()
        self.metadata_extractor = MetadataExtractor()

    def parse_job_description(self, jd_input: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse raw text or JSON Job Description to extract title, required skills, and required experience.
        """
        if isinstance(jd_input, dict):
            description_text = jd_input.get("description", "")
            title = jd_input.get("title", "Job Role")
            must_have_skills = jd_input.get("must_have_skills", [])
            req_years = float(jd_input.get("required_experience_years", 0))
        else:
            description_text = str(jd_input)
            title = "Job Role"
            must_have_skills = []
            req_years = 0.0

        # Extract skills if not provided
        if not must_have_skills:
            must_have_skills = self.metadata_extractor.extract_skills(description_text)

        # Extract required years if not provided
        if req_years == 0.0:
            req_years = self.metadata_extractor.extract_years_experience(description_text)

        return {
            "title": title,
            "description": description_text,
            "must_have_skills": must_have_skills,
            "required_experience_years": req_years
        }

    def match(
        self,
        jd_input: Union[str, Dict[str, Any]],
        top_k: int = 10,
        semantic_weight: float = 0.50,
        bm25_weight: float = 0.30,
        exp_weight: float = 0.20
    ) -> Dict[str, Any]:
        """
        Perform hybrid matching of job description against indexed resumes.
        Returns exact target JSON structure.
        """
        parsed_jd = self.parse_job_description(jd_input)
        jd_text = parsed_jd["description"]
        must_have_skills = [s.lower() for s in parsed_jd["must_have_skills"]]
        req_exp = parsed_jd["required_experience_years"]

        # 1. Retrieve all candidate chunks from Vector DB
        all_records = self.vector_store.get_all_records()
        documents = all_records.get("documents", [])
        metadatas = all_records.get("metadatas", [])

        if not documents:
            return {
                "job_description": jd_text,
                "top_matches": []
            }

        # 2. Vector Semantic Similarity Search
        vector_results = self.vector_store.search_vector(jd_text, top_k=min(100, len(documents)))
        vec_docs = vector_results.get("documents", [[]])[0]
        vec_metas = vector_results.get("metadatas", [[]])[0]
        vec_distances = vector_results.get("distances", [[]])[0]

        # Convert cosine distance to cosine similarity (range 0 to 1)
        # Cosine distance in ChromaDB cosine space is 1 - cosine_similarity
        vec_sim_map = {}
        for idx, meta in enumerate(vec_metas):
            doc_id = meta.get("resume_path", "") + "_" + str(meta.get("chunk_index", 0))
            dist = vec_distances[idx] if idx < len(vec_distances) else 1.0
            similarity = max(0.0, 1.0 - dist)
            vec_sim_map[doc_id] = similarity

        # 3. BM25 Keyword Matching over all candidate documents
        tokenized_corpus = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        jd_tokens = jd_text.lower().split()
        bm25_scores = bm25.get_scores(jd_tokens)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        normalized_bm25 = [s / max_bm25 for s in bm25_scores]

        # 4. Group & Aggregate Scores per Candidate (by resume_path)
        candidate_map = {}

        for i, meta in enumerate(metadatas):
            path = meta.get("resume_path", meta.get("filename", "Unknown"))
            c_name = meta.get("candidate_name", "Candidate")
            c_skills = [s.strip() for s in meta.get("skills", "").split(",") if s.strip()]
            c_skills_lower = [s.lower() for s in c_skills]
            c_exp = float(meta.get("years_of_experience", 0.0))

            doc_id = path + "_" + str(meta.get("chunk_index", 0))
            vec_sim = vec_sim_map.get(doc_id, 0.2)
            bm25_sim = normalized_bm25[i]

            chunk_text = documents[i]

            if path not in candidate_map:
                candidate_map[path] = {
                    "candidate_name": c_name,
                    "resume_path": path,
                    "skills": c_skills,
                    "skills_lower": c_skills_lower,
                    "years_of_experience": c_exp,
                    "chunks": [],
                    "vec_sims": [],
                    "bm25_sims": []
                }

            candidate_map[path]["chunks"].append((chunk_text, vec_sim, meta.get("section", "GENERAL")))
            candidate_map[path]["vec_sims"].append(vec_sim)
            candidate_map[path]["bm25_sims"].append(bm25_sim)

        # 5. Score and Filter Candidates
        scored_candidates = []

        for path, cand in candidate_map.items():
            max_vec_sim = max(cand["vec_sims"]) if cand["vec_sims"] else 0.0
            avg_bm25_sim = float(np.mean(cand["bm25_sims"])) if cand["bm25_sims"] else 0.0

            # Skill overlap score
            matched_skills = [s for s in cand["skills"] if s.lower() in must_have_skills]
            skill_ratio = len(matched_skills) / max(1, len(must_have_skills))

            # Experience match ratio
            if req_exp > 0:
                exp_ratio = min(1.0, cand["years_of_experience"] / req_exp)
            else:
                exp_ratio = 1.0

            # Raw weighted score 0 to 1
            combined_raw = (
                semantic_weight * max_vec_sim +
                bm25_weight * (0.5 * avg_bm25_sim + 0.5 * skill_ratio) +
                exp_weight * exp_ratio
            )
            match_score = int(round(combined_raw * 100))
            match_score = max(0, min(100, match_score))

            # Hard filtering / Penalization for missing must-have requirements
            penalty_applied = False
            missing_skills = [s for s in must_have_skills if s not in cand["skills_lower"]]
            
            # If missing more than half must-have skills, apply penalty
            if len(must_have_skills) >= 2 and len(missing_skills) > len(must_have_skills) / 2:
                match_score = int(match_score * 0.6)
                penalty_applied = True

            # If experience significantly below required years, apply penalty
            if req_exp >= 3 and cand["years_of_experience"] < req_exp - 1.5:
                match_score = int(match_score * 0.7)
                penalty_applied = True

            # Extract top relevant excerpts (up to 3 best vector matching chunks)
            sorted_chunks = sorted(cand["chunks"], key=lambda x: x[1], reverse=True)
            excerpts = [c[0] for c in sorted_chunks[:3]]

            # Build reasoning explanation
            reasons = []
            if matched_skills:
                reasons.append(f"Matched key skills: {', '.join(matched_skills[:5])}.")
            if cand["years_of_experience"] > 0:
                reasons.append(f"Candidate has ~{cand['years_of_experience']} years of relevant experience (Required: {req_exp} yrs).")
            if sorted_chunks:
                top_sec = sorted_chunks[0][2]
                reasons.append(f"Strongest semantic match in [{top_sec}] section.")
            if penalty_applied:
                reasons.append("Score penalized due to missing must-have skills or insufficient experience years.")

            reasoning = " ".join(reasons)

            scored_candidates.append({
                "candidate_name": cand["candidate_name"],
                "resume_path": cand["resume_path"],
                "match_score": match_score,
                "matched_skills": matched_skills if matched_skills else cand["skills"][:5],
                "relevant_excerpts": excerpts,
                "reasoning": reasoning
            })

        # 6. Sort candidates by match_score descending and slice top_k
        scored_candidates.sort(key=lambda x: x["match_score"], reverse=True)
        top_matches = scored_candidates[:top_k]

        return {
            "job_description": jd_text,
            "top_matches": top_matches
        }


if __name__ == "__main__":
    matcher = JobMatcher()
    sample_jd = {
        "title": "Senior Python Developer",
        "description": "Looking for a Senior Python Developer with 4+ years experience in Django, FastAPI, REST APIs, and PostgreSQL.",
        "must_have_skills": ["Python", "Django", "FastAPI", "PostgreSQL"],
        "required_experience_years": 4
    }
    result = matcher.match(sample_jd, top_k=5)
    print(json.dumps(result, indent=2))
