import os
import json
from datasets import load_dataset

def save_sample(dataset, name, split='train'):
    try:
        sample = [dict(x) for x in dataset[split].select(range(min(10, len(dataset[split]))))]
        os.makedirs(f"datasets/{name}", exist_ok=True)
        with open(f"datasets/{name}/samples.json", "w") as f:
            json.dump(sample, f, indent=2, default=str)
        print(f"Saved sample for {name}")
    except Exception as e:
        print(f"Error saving sample for {name}: {e}")

print("Downloading MMLU...")
try:
    # Use 'abstract_algebra' as 'all' might be too huge to download all at once just for resource finding
    # Actually 'all' is fine if we stream or just download one config.
    # Let's download 'abstract_algebra' as a representative sample.
    mmlu = load_dataset("cais/mmlu", "abstract_algebra")
    mmlu.save_to_disk("datasets/mmlu_abstract_algebra")
    save_sample(mmlu, "mmlu_abstract_algebra", split='test') # MMLU usually has test
except Exception as e:
    print(f"Error downloading MMLU: {e}")

print("Downloading XNLI...")
try:
    xnli = load_dataset("xnli", "all_languages")
    # Save a small subset to disk to save space/time
    xnli_small = xnli['test'].select(range(100))
    xnli_small.save_to_disk("datasets/xnli_sample")
    save_sample(xnli, "xnli_sample", split='test')
except Exception as e:
    print(f"Error downloading XNLI: {e}")

print("Downloading FLORES...")
try:
    # Try openlanguagedata/flores_plus
    # It likely needs config. Let's try to list configs or pick a pair.
    # Actually, let's use Muennighoff/flores200 as it might be easier.
    # Or just `facebook/flores` with `trust_remote_code=True` if we were allowed.
    # I'll try `openlanguagedata/flores_plus` with a default config if it exists.
    # If not, I'll try to load 'eng_Latn-zho_Hans' (English to Chinese)
    flores = load_dataset("openlanguagedata/flores_plus", "eng_Latn-zho_Hans", trust_remote_code=True)
    flores.save_to_disk("datasets/flores_en_zh")
    save_sample(flores, "flores_en_zh", split='dev')
except Exception as e:
    print(f"Error downloading FLORES: {e}")
