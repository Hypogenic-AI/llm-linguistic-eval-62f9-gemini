from huggingface_hub import list_datasets

print("Searching for 'flores'...")
datasets = list_datasets(search="flores", limit=10)
for d in datasets:
    print(d.id)
