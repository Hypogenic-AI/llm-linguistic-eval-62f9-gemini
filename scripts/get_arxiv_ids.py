import requests
import re
import time

def get_arxiv_id(title):
    query = title.replace(" ", "+")
    url = f"https://arxiv.org/search/?query={query}&searchtype=title&source=header"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Look for arXiv:XXXX.XXXXX
            match = re.search(r'arXiv:(\d+\.\d+)', response.text)
            if match:
                return match.group(1)
    except Exception as e:
        print(f"Error searching for {title}: {e}")
    return None

papers = [
    "Is Translation All You Need?",
]

print("Searching for remaining...")
found_papers = []
for title in papers:
    print(f"Searching: {title}")
    arxiv_id = get_arxiv_id(title)
    if arxiv_id:
        print(f"  Found: {arxiv_id}")
        found_papers.append((title, arxiv_id))
    else:
        print("  Not found")
    time.sleep(1)

print("\nResults:")
for title, aid in found_papers:
    print(f"{aid} | {title}")

with open("found_ids.txt", "a") as f:
    for title, aid in found_papers:
        f.write(f"{aid}|{title}\n")