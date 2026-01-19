import datasets
from datasets import load_dataset_builder

target_datasets = [
    "facebook/flores",
    "cais/mmlu",
    "xnli"
]

print("Checking datasets on Hugging Face...")
for name in target_datasets:
    try:
        builder = load_dataset_builder(name)
        print(f"✅ {name} found.")
        # print(f"  Description: {builder.info.description[:100]}...")
    except Exception as e:
        print(f"❌ {name} not found or error: {e}")
