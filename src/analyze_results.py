import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_results():
    files = glob.glob("results/raw/*.json")
    all_data = []
    for f in files:
        try:
            with open(f, 'r') as fd:
                data = json.load(fd)
                print(f"Loaded {f}: type={type(data)}, len={len(data) if hasattr(data, '__len__') else 'N/A'}")
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"Warning: {f} is not a list!")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return pd.DataFrame(all_data)

def compute_metrics(df):
    # Separate English results to use as baseline
    en_df = df[df['lang'] == 'en'].set_index('id')
    
    metrics = []
    
    languages = df['lang'].unique()
    for lang in languages:
        lang_df = df[df['lang'] == lang].set_index('id')
        
        # Intersection of IDs (should be all, but safe to check)
        common_ids = lang_df.index.intersection(en_df.index)
        
        # Accuracy
        acc = lang_df['correct'].mean()
        
        # Consistency with English (only for common IDs)
        if lang != 'en':
            lang_preds = lang_df.loc[common_ids, 'prediction']
            en_preds = en_df.loc[common_ids, 'prediction']
            truth = lang_df.loc[common_ids, 'ground_truth']
            
            # Agreement
            agreement = (lang_preds == en_preds).mean()
            
            # Same Mistake (Both wrong and same prediction)
            both_wrong = (~(lang_preds == truth)) & (~(en_preds == truth))
            same_mistake = (lang_preds[both_wrong] == en_preds[both_wrong]).mean() if both_wrong.sum() > 0 else 0
            
            # Deviation (Lang wrong, En correct)
            lang_wrong_en_correct = ((~(lang_preds == truth)) & (en_preds == truth)).mean()
            
            # Better (Lang correct, En wrong)
            lang_correct_en_wrong = ((lang_preds == truth) & (~(en_preds == truth))).mean()
            
        else:
            agreement = 1.0
            same_mistake = 0.0
            lang_wrong_en_correct = 0.0
            lang_correct_en_wrong = 0.0
            
        metrics.append({
            'lang': lang,
            'accuracy': acc,
            'agreement_with_en': agreement,
            'same_mistake_ratio': same_mistake,
            'samples': len(lang_df)
        })
        
    return pd.DataFrame(metrics)

def plot_results(metrics_df):
    os.makedirs("results/plots", exist_ok=True)
    
    # Sort by resource level roughly
    # High: en, es, fr, zh
    # Mid: vi, ko, id, ar
    # Low: bn, te, sw, ta
    
    order = ['en', 'es', 'fr', 'zh', 'vi', 'ko', 'id', 'ar', 'bn', 'te', 'sw', 'ta']
    # Filter only present langs
    order = [l for l in order if l in metrics_df['lang'].values]
    
    metrics_df = metrics_df.set_index('lang').reindex(order).reset_index()
    
    # 1. Accuracy Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='lang', y='accuracy', palette='viridis')
    plt.axhline(y=metrics_df[metrics_df['lang']=='en']['accuracy'].values[0], color='r', linestyle='--', label='English Baseline')
    plt.title('Accuracy by Language (GPT-4o)')
    plt.ylabel('Accuracy')
    plt.xlabel('Language')
    plt.legend()
    plt.savefig('results/plots/accuracy_by_lang.png')
    
    # 2. Agreement with English
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df[metrics_df['lang']!='en'], x='lang', y='agreement_with_en', palette='magma')
    plt.title('Prediction Agreement with English')
    plt.ylabel('Agreement Ratio')
    plt.xlabel('Language')
    plt.savefig('results/plots/agreement_with_en.png')
    
    # 3. Scatter: Accuracy vs Agreement
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=metrics_df[metrics_df['lang']!='en'], x='accuracy', y='agreement_with_en', hue='lang', s=100)
    plt.title('Accuracy vs Agreement with English')
    plt.xlabel('Accuracy')
    plt.ylabel('Agreement with English')
    
    # Add labels
    for i, row in metrics_df[metrics_df['lang']!='en'].iterrows():
        plt.text(row['accuracy']+0.005, row['agreement_with_en'], row['lang'])
        
    plt.savefig('results/plots/acc_vs_agreement.png')

def main():
    df = load_results()
    metrics = compute_metrics(df)
    print("Metrics Summary:")
    print(metrics)
    
    metrics.to_csv("results/metrics_summary.csv", index=False)
    plot_results(metrics)

if __name__ == "__main__":
    main()
