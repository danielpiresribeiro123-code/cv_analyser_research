from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..fast_analysis.modules.parse_pdf import get_full_resume_text


# carregar modelo de embeddings (leve e offline)
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_semantic_ranking(cvs_dir: str, job_description: str):
    cvs_path = Path(cvs_dir)

    # embedding da job description
    job_emb = model.encode([job_description])

    ranking = []

    for pdf_file in cvs_path.glob("*.pdf"):
        text = get_full_resume_text(str(pdf_file))

        cv_emb = model.encode([text])

        sim = cosine_similarity(cv_emb, job_emb)[0][0]

        ranking.append({
            "file": pdf_file.name,
            "semantic_score": float(sim)
        })

    # ordenar melhor → pior
    ranking.sort(key=lambda x: x["semantic_score"], reverse=True)

    return ranking
