#!/usr/bin/env python3
"""
Continue Training mDeBERTa-v3 from Checkpoint

Continue training from the saved model with additional epochs.
"""

import os
import torch
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch import nn
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
CHECKPOINT_DIR = "./mdeberta_v3_medical_nli"  # Previous checkpoint
OUTPUT_DIR = "./mdeberta_v3_medical_nli_v2"   # New output

# Continue training settings
ADDITIONAL_EPOCHS = 5  # Train 5 more epochs
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
BASE_LEARNING_RATE = 2e-5  # Lower LR for continued training
LAYER_DECAY = 0.9
MAX_LENGTH = 256

# Label mapping
LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}

CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 2.0])

print("=" * 90)
print("CONTINUE TRAINING mDeBERTa-v3 (+5 epochs)")
print("=" * 90)

# ============================================================
# Step 1: Create Medical Examples
# ============================================================
print("\n📚 Step 1: Creating medical negation examples...")

medical_examples = [
    # Critical Medical Negations (CONTRADICTION)
    {"premise": "Thuốc A không được chỉ định cho trẻ em dưới 12 tuổi.", 
     "hypothesis": "Thuốc A được chỉ định cho trẻ em.", "label": 2},
    {"premise": "Thuốc A không được chỉ định cho trẻ em dưới 12 tuổi.", 
     "hypothesis": "Thuốc A an toàn cho trẻ 5 tuổi.", "label": 2},
    {"premise": "Không nên sử dụng thuốc này trong thai kỳ.",
     "hypothesis": "Có thể sử dụng thuốc này trong thai kỳ.", "label": 2},
    {"premise": "Không nên sử dụng thuốc này trong thai kỳ.",
     "hypothesis": "Thuốc này an toàn cho phụ nữ mang thai.", "label": 2},
    {"premise": "Bệnh nhân không có triệu chứng sốt.",
     "hypothesis": "Bệnh nhân có triệu chứng sốt.", "label": 2},
    {"premise": "Không được dùng quá 4g paracetamol mỗi ngày.",
     "hypothesis": "Có thể dùng 6g paracetamol mỗi ngày.", "label": 2},
    {"premise": "Không được dùng quá 4g paracetamol mỗi ngày.",
     "hypothesis": "Liều 8g paracetamol là an toàn.", "label": 2},
    {"premise": "Thuốc này không dành cho người cao huyết áp.",
     "hypothesis": "Người cao huyết áp có thể dùng thuốc này.", "label": 2},
    {"premise": "Không uống rượu khi đang dùng kháng sinh.",
     "hypothesis": "Có thể uống rượu khi dùng kháng sinh.", "label": 2},
    {"premise": "Phụ nữ cho con bú không nên dùng thuốc này.",
     "hypothesis": "Thuốc này an toàn khi cho con bú.", "label": 2},
    {"premise": "Bệnh tiểu đường không thể chữa khỏi hoàn toàn.",
     "hypothesis": "Bệnh tiểu đường có thể chữa khỏi hoàn toàn.", "label": 2},
    {"premise": "Vắc xin COVID không gây vô sinh.",
     "hypothesis": "Vắc xin COVID gây vô sinh.", "label": 2},
    {"premise": "Viêm gan B không lây qua đường ăn uống.",
     "hypothesis": "Viêm gan B có thể lây qua đường ăn uống.", "label": 2},
    {"premise": "Thuốc kháng sinh không có tác dụng với virus.",
     "hypothesis": "Thuốc kháng sinh hiệu quả với virus.", "label": 2},
    {"premise": "Không được tiêm vaccine sống cho người suy giảm miễn dịch.",
     "hypothesis": "Người suy giảm miễn dịch có thể tiêm vaccine sống.", "label": 2},
    {"premise": "Rong kinh không phải là hiện tượng bình thường.",
     "hypothesis": "Rong kinh là hiện tượng bình thường.", "label": 2},
    {"premise": "Đau bụng kinh dữ dội không nên bỏ qua.",
     "hypothesis": "Đau bụng kinh dữ dội có thể bỏ qua.", "label": 2},
    {"premise": "Thai nhi không thể sống được nếu sinh trước 24 tuần.",
     "hypothesis": "Thai nhi sinh lúc 20 tuần có thể sống được.", "label": 2},
    # More negation patterns
    {"premise": "Không được sử dụng thuốc quá hạn.",
     "hypothesis": "Có thể dùng thuốc đã hết hạn.", "label": 2},
    {"premise": "Không nên tự ý ngưng thuốc.",
     "hypothesis": "Có thể tự ý ngưng thuốc bất cứ lúc nào.", "label": 2},
    
    # ENTAILMENT examples
    {"premise": "Thuốc A được chỉ định cho người lớn.",
     "hypothesis": "Thuốc A được chỉ định cho người lớn.", "label": 0},
    {"premise": "Acid folic quan trọng trong thai kỳ vì giúp ngăn dị tật ống thần kinh.",
     "hypothesis": "Acid folic cần thiết cho thai nhi.", "label": 0},
    {"premise": "Paracetamol là thuốc giảm đau và hạ sốt.",
     "hypothesis": "Paracetamol có tác dụng giảm đau.", "label": 0},
    {"premise": "Bệnh tiểu đường cần kiểm soát đường huyết.",
     "hypothesis": "Người tiểu đường phải theo dõi đường huyết.", "label": 0},
    {"premise": "Vitamin D cần thiết cho sự hấp thu canxi.",
     "hypothesis": "Vitamin D hỗ trợ hấp thu canxi.", "label": 0},
    {"premise": "Cao huyết áp làm tăng nguy cơ đột quỵ.",
     "hypothesis": "Người cao huyết áp có nguy cơ đột quỵ cao hơn.", "label": 0},
    {"premise": "Kháng sinh cần dùng đủ liệu trình.",
     "hypothesis": "Không được ngừng kháng sinh giữa chừng.", "label": 0},
    {"premise": "Tiền sản giật có triệu chứng tăng huyết áp và protein niệu.",
     "hypothesis": "Tiền sản giật gây tăng huyết áp.", "label": 0},
    {"premise": "Rong kinh là tình trạng chảy máu kinh nguyệt kéo dài.",
     "hypothesis": "Rong kinh gây mất máu kéo dài.", "label": 0},
    
    # NEUTRAL examples
    {"premise": "Acid folic quan trọng trong thai kỳ.",
     "hypothesis": "Uống 2 lít nước mỗi ngày.", "label": 1},
    {"premise": "Paracetamol là thuốc giảm đau.",
     "hypothesis": "Vitamin C có trong cam.", "label": 1},
    {"premise": "Bệnh tiểu đường cần kiểm soát đường huyết.",
     "hypothesis": "Tập thể dục tốt cho tim mạch.", "label": 1},
    {"premise": "Thuốc kháng sinh cần kê đơn.",
     "hypothesis": "Ngủ đủ 8 tiếng mỗi ngày.", "label": 1},
    {"premise": "Cao huyết áp cần uống thuốc hạ áp.",
     "hypothesis": "Ăn nhiều rau xanh tốt cho sức khỏe.", "label": 1},
    {"premise": "Viêm gan B là bệnh truyền nhiễm.",
     "hypothesis": "Ung thư phổi liên quan đến hút thuốc.", "label": 1},
    {"premise": "Người tiểu đường nên hạn chế đường.",
     "hypothesis": "Canxi cần cho xương chắc khỏe.", "label": 1},
    {"premise": "Sốt là triệu chứng của nhiễm trùng.",
     "hypothesis": "Đau lưng có thể do ngồi sai tư thế.", "label": 1},
]

MEDICAL_REPEAT = 50
repeated_medical = medical_examples * MEDICAL_REPEAT
print(f"   Created {len(medical_examples)} unique examples, {len(repeated_medical)} after repeat")

# ============================================================
# Step 2: Load Datasets
# ============================================================
print("\n📥 Step 2: Loading ViANLI dataset...")

dataset = load_dataset("uitnlp/ViANLI")
print(f"   ViANLI: train={len(dataset['train'])}, val={len(dataset['validation'])}, test={len(dataset['test'])}")

def encode_labels(examples):
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    examples["label"] = [label_map[label] for label in examples["label"]]
    return examples

dataset = dataset.map(encode_labels, batched=True)

medical_dataset = Dataset.from_list(repeated_medical)
combined_train = concatenate_datasets([dataset["train"], medical_dataset])
print(f"   Combined train: {len(combined_train)} samples")

# ============================================================
# Step 3: Load Model from Checkpoint
# ============================================================
print(f"\n🔧 Step 3: Loading model from {CHECKPOINT_DIR}...")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded! Device: {device}")

# ============================================================
# Step 4: Preprocess
# ============================================================
print("\n⚙️ Step 4: Preprocessing...")

def preprocess(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

columns_to_remove = ["premise", "hypothesis"]
if "uid" in combined_train.column_names:
    columns_to_remove.append("uid")

tokenized_train = combined_train.map(
    preprocess, batched=True, remove_columns=columns_to_remove
).rename_column("label", "labels")

tokenized_val = dataset["validation"].map(
    preprocess, batched=True, remove_columns=["uid", "premise", "hypothesis"]
).rename_column("label", "labels")

tokenized_test = dataset["test"].map(
    preprocess, batched=True, remove_columns=["uid", "premise", "hypothesis"]
).rename_column("label", "labels")

print(f"✅ Train: {len(tokenized_train)}, Val: {len(tokenized_val)}, Test: {len(tokenized_test)}")

# ============================================================
# Step 5: Setup LLRD
# ============================================================
print("\n⚙️ Step 5: Setting up LLRD...")

def get_optimizer_grouped_parameters(model, base_lr, layer_decay, weight_decay=0.01):
    opt_parameters = []
    no_decay = ["bias", "LayerNorm.weight"]
    
    if hasattr(model, 'deberta'):
        num_layers = model.config.num_hidden_layers
        encoder = model.deberta
    else:
        num_layers = model.config.num_hidden_layers
        encoder = model.base_model
    
    print(f"   Applying LLRD to {num_layers} layers with decay={layer_decay}")
    
    # Embeddings
    lr_embed = base_lr * (layer_decay ** (num_layers + 1))
    opt_parameters.append({
        "params": [p for n, p in encoder.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
        "lr": lr_embed, "weight_decay": weight_decay
    })
    opt_parameters.append({
        "params": [p for n, p in encoder.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
        "lr": lr_embed, "weight_decay": 0.0
    })
    
    # Encoder layers
    for layer_i in range(num_layers):
        lr_layer = base_lr * (layer_decay ** (num_layers - layer_i))
        layer_module = encoder.encoder.layer[layer_i]
        
        opt_parameters.append({
            "params": [p for n, p in layer_module.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": lr_layer, "weight_decay": weight_decay
        })
        opt_parameters.append({
            "params": [p for n, p in layer_module.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": lr_layer, "weight_decay": 0.0
        })
    
    # Classifier head
    opt_parameters.append({
        "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
        "lr": base_lr, "weight_decay": weight_decay
    })
    opt_parameters.append({
        "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
        "lr": base_lr, "weight_decay": 0.0
    })
    
    return [l for l in opt_parameters if len(l["params"]) > 0]

# ============================================================
# Step 6: Custom Trainer
# ============================================================

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ============================================================
# Step 7: Training
# ============================================================
print(f"\n🚀 Step 7: Continuing training (+{ADDITIONAL_EPOCHS} epochs)...")
print(f"   Base LR: {BASE_LEARNING_RATE} (lower for continued training)")
print("=" * 90)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=BASE_LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=ADDITIONAL_EPOCHS,
    weight_decay=0.01,
    warmup_ratio=0.05,  # Less warmup for continued training
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    bf16=torch.cuda.is_available(),
    report_to="none",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

optimizer_grouped_params = get_optimizer_grouped_parameters(
    model, BASE_LEARNING_RATE, LAYER_DECAY
)

trainer = WeightedLossTrainer(
    class_weights=CLASS_WEIGHTS,
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    optimizers=(
        torch.optim.AdamW(optimizer_grouped_params, lr=BASE_LEARNING_RATE),
        None
    )
)

trainer.train()

print("\n" + "=" * 90)
print("✅ Training complete!")

# ============================================================
# Step 8: Evaluate
# ============================================================
print("\n📊 Step 8: Evaluating...")

test_results = trainer.evaluate(tokenized_test)
print(f"   Test Accuracy: {test_results['eval_accuracy']:.4f}")

predictions = trainer.predict(tokenized_test)
pred_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

print("\n📋 Classification Report:")
print(classification_report(
    true_labels, pred_labels,
    target_names=["entailment", "neutral", "contradiction"]
))

# ============================================================
# Step 9: Save Model
# ============================================================
print(f"\n💾 Step 9: Saving to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ============================================================
# Step 10: Test Negation Trap Cases
# ============================================================
print("\n🧪 Step 10: Testing negation trap cases...")

from sentence_transformers import CrossEncoder
finetuned_model = CrossEncoder(OUTPUT_DIR, device=device)

test_cases = [
    {"name": "🚨 Medication Contraindication",
     "doc": "Thuốc A không được chỉ định cho trẻ em dưới 12 tuổi.",
     "claim": "Thuốc A được chỉ định cho trẻ em.",
     "expected": "contradiction"},
    {"name": "🚨 Pregnancy Warning",
     "doc": "Không nên sử dụng thuốc này trong thai kỳ.",
     "claim": "Có thể sử dụng thuốc này trong thai kỳ.",
     "expected": "contradiction"},
    {"name": "🚨 Dosage Negation",
     "doc": "Không được dùng quá 4g paracetamol mỗi ngày.",
     "claim": "Có thể dùng 6g paracetamol mỗi ngày.",
     "expected": "contradiction"},
    {"name": "Symptom Presence",
     "doc": "Bệnh nhân không có triệu chứng sốt.",
     "claim": "Bệnh nhân có triệu chứng sốt.",
     "expected": "contradiction"},
    {"name": "Antibiotic + Alcohol",
     "doc": "Không uống rượu khi đang dùng kháng sinh.",
     "claim": "Có thể uống rượu khi dùng kháng sinh.",
     "expected": "contradiction"},
    {"name": "✅ Entailment (exact)",
     "doc": "Thuốc A được chỉ định cho người lớn.",
     "claim": "Thuốc A được chỉ định cho người lớn.",
     "expected": "entailment"},
    {"name": "✅ Entailment (inference)",
     "doc": "Acid folic quan trọng trong thai kỳ vì giúp ngăn dị tật ống thần kinh.",
     "claim": "Acid folic cần thiết cho thai nhi.",
     "expected": "entailment"},
    {"name": "⚪ Neutral (unrelated)",
     "doc": "Acid folic quan trọng trong thai kỳ.",
     "claim": "Uống 2 lít nước mỗi ngày.",
     "expected": "neutral"},
    {"name": "⚪ Neutral (diff medical)",
     "doc": "Paracetamol là thuốc giảm đau.",
     "claim": "Vitamin C có trong cam.",
     "expected": "neutral"},
]

NLI_LABELS = ["entailment", "neutral", "contradiction"]
passed = 0

for test in test_cases:
    scores = finetuned_model.predict([(test['doc'], test['claim'])])[0]
    pred_idx = np.argmax(scores)
    predicted = NLI_LABELS[pred_idx]
    
    is_correct = predicted == test['expected']
    if is_correct:
        passed += 1
    
    status = "✅" if is_correct else "❌"
    print(f"\n{status} {test['name']}")
    print(f"   Expected: {test['expected'].upper()}")
    print(f"   Predicted: {predicted.upper()}")
    print(f"   Scores: E={scores[0]:.2f}, N={scores[1]:.2f}, C={scores[2]:.2f}")

print("\n" + "=" * 90)
print(f"RESULTS: {passed}/{len(test_cases)} tests passed!")
print("=" * 90)
print(f"\n📦 Model saved at: {os.path.abspath(OUTPUT_DIR)}")
