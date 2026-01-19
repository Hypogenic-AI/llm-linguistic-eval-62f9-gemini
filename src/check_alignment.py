from datasets import load_dataset

def check_alignment():
    ds_en = load_dataset("aialt/MuBench", "MMLUDataset_en_template_en", split="test", streaming=True)
    ds_zh = load_dataset("aialt/MuBench", "MMLUDataset_en_template_zh", split="test", streaming=True)
    
    iter_en = iter(ds_en)
    iter_zh = iter(ds_zh)
    
    print("Checking label alignment for first 5 items...")
    for i in range(5):
        item_en = next(iter_en)
        item_zh = next(iter_zh)
        
        print(f"ID: {item_en['_id']} | En Label: {item_en['label']} | Zh Label: {item_zh['label']}")
        if item_en['_id'] != item_zh['_id']:
            print("MISMATCH IN ID!")
        if item_en['label'] != item_zh['label']:
            print("MISMATCH IN LABEL!")

if __name__ == "__main__":
    check_alignment()
