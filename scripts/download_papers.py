import os
import time

papers = [
    ("2404.18534", "evaluating_linguistic_discrimination.pdf"),
    ("2504.04155", "gloteval.pdf"),
    ("2506.19468", "mubench.pdf"),
    ("2404.00929", "multilingual_llm_survey.pdf"),
    ("2402.18815", "how_llms_handle_multilingualism.pdf")
]

os.makedirs("papers", exist_ok=True)

for arxiv_id, filename in papers:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"Downloading {filename} from {url}...")
    os.system(f"wget -q {url} -O papers/{filename}")
    time.sleep(1)

print("Downloads complete.")
