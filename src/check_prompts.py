from datasets import load_dataset

def check_zh_prompt():
    ds_zh = load_dataset("aialt/MuBench", "MMLUDataset_en_template_zh", split="test", streaming=True)
    item = next(iter(ds_zh))
    print("Chinese Prompt Example:")
    print(item['prompt'])

if __name__ == "__main__":
    check_zh_prompt()
