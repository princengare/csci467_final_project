import os
import torch
import numpy as np
import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.linear_model import Ridge
from datasets import Dataset
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
use_fp16 = torch.cuda.is_available()
print("Using GPU:", use_fp16)

os.makedirs("results", exist_ok=True)
results_log_path = "results/summary.csv"
all_results = []

print("Loading datasets...")
dfs = [pd.read_csv(f"data/{name}.csv") for name in [
    "poetry_foundation", "genius_lyrics", "multilingual_lit", "twitter_data", "opus_dialogue"
]]
full_df = pd.concat(dfs, ignore_index=True).dropna(subset=['text', 'label'])

train_df, temp_df = train_test_split(full_df, test_size=0.3, stratify=full_df['label'], random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print("Running rule-based classifier...")
keywords = {
    'romantic': ["love", "passion", "kiss", "heart", "desire", "darling"],
    'familial': ["mother", "father", "support", "care", "brother", "sister"],
    'platonic': ["friend", "companionship", "bond", "connection", "respect"]
}
rb_preds = []
for text in test_df['text']:
    text_lower = text.lower()
    scores = {label: sum(kw in text_lower for kw in kw_list) for label, kw_list in keywords.items()}
    pred = max(scores, key=scores.get) if any(scores.values()) else 'romantic'
    rb_preds.append(pred)

rb_report = classification_report(test_df['label'], rb_preds, output_dict=True, zero_division=0)
print("Rule-Based Classifier Results:", rb_report)
all_results.append({"model": "Rule-Based", "accuracy": rb_report['accuracy'], "f1": rb_report['macro avg']['f1-score']})

tokenizer = XLMRobertaTokenizer.from_pretrained("facebook/xlm-roberta-small")
label2id = {label: i for i, label in enumerate(sorted(train_df['label'].unique()))}
id2label = {i: label for label, i in label2id.items()}

def preprocess(example):
    enc = tokenizer(example['text'], truncation=True, padding='max_length', max_length=128)
    enc['labels'] = [label2id[label] for label in example['label']]
    return enc

train_dataset = Dataset.from_pandas(train_df).map(preprocess, batched=True)
dev_dataset = Dataset.from_pandas(dev_df).map(preprocess, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(preprocess, batched=True)

columns = ['input_ids', 'attention_mask', 'labels']
for ds in [train_dataset, dev_dataset, test_dataset]:
    ds.set_format(type='torch', columns=columns)

print("Evaluating frozen XLM-R (small)...")
frozen_model = XLMRobertaForSequenceClassification.from_pretrained(
    "facebook/xlm-roberta-small", num_labels=len(label2id), problem_type="single_label_classification"
)
for param in frozen_model.roberta.parameters():
    param.requires_grad = False

trainer_frozen = Trainer(
    model=frozen_model,
    args=TrainingArguments(
        output_dir="results/frozen",
        evaluation_strategy="epoch",
        save_strategy="no",
        per_device_eval_batch_size=64,
        logging_dir="logs/frozen",
        report_to="none",
        fp16=use_fp16
    ),
    eval_dataset=dev_dataset
)

print("Skipping training for frozen model (zero-shot evaluation)...")
outputs = trainer_frozen.predict(test_dataset)
preds = np.argmax(outputs.predictions, axis=1)
labels = outputs.label_ids

f1 = classification_report(labels, preds, output_dict=True, zero_division=0)['macro avg']['f1-score']
acc = (preds == labels).mean()
print("Frozen XLM-R Prediction Accuracy:", acc)

all_results.append({"model": "XLM-R Small Frozen", "accuracy": acc, "f1": f1})

print("Training fine-tuned XLM-R (small)...")
fine_tune_model = XLMRobertaForSequenceClassification.from_pretrained(
    "facebook/xlm-roberta-small", num_labels=len(label2id), problem_type="single_label_classification"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1 = classification_report(labels, preds, output_dict=True, zero_division=0)['macro avg']['f1-score']
    return {"accuracy": (preds == labels).mean(), "f1": f1}

trainer = Trainer(
    model=fine_tune_model,
    args=TrainingArguments(
        output_dir="results/finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        logging_dir="logs/finetuned",
        report_to="none",
        fp16=use_fp16
    ),
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
results = trainer.evaluate(test_dataset)
print("Fine-tuned XLM-R (small) Test Results:", results)
all_results.append({"model": "XLM-R Small Fine-Tuned", "accuracy": results['eval_accuracy'], "f1": results['eval_f1']})

print("Running VAD regression...")
def extract_embeddings(texts):
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = fine_tune_model.roberta(**encodings)
        return outputs.last_hidden_state.mean(dim=1).numpy()

train_embs = extract_embeddings(list(train_df['text']))
test_embs = extract_embeddings(list(test_df['text']))

reg_v = Ridge().fit(train_embs, train_df.get("valence", 0))
reg_a = Ridge().fit(train_embs, train_df.get("arousal", 0))
reg_d = Ridge().fit(train_embs, train_df.get("dominance", 0))

vad_preds = {
    "valence": reg_v.predict(test_embs),
    "arousal": reg_a.predict(test_embs),
    "dominance": reg_d.predict(test_embs)
}

vad_mae = {k: mean_absolute_error(test_df[k], v) for k, v in vad_preds.items()}
print("VAD MAE:", vad_mae)
all_results.append({"model": "VAD Regression", "accuracy": "NA", "f1": "NA", **vad_mae})

summary_df = pd.DataFrame(all_results)
summary_df.to_csv(results_log_path, index=False)
print("\n Results summary saved to", results_log_path)
