import os

papers = [
    "gloteval.pdf",
    "mubench.pdf"
]

for filename in papers:
    path = os.path.join("papers", filename)
    print(f"=== {filename} ===")
    try:
        # Use simple os.system to cat the file to stdout is not good for binary
        # I will rely on read_file tool again, as it worked perfectly.
        pass
    except Exception as e:
        print(f"Error reading {filename}: {e}")
