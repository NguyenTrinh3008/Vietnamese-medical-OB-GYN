#!/usr/bin/env python3
"""
NLI Hallucination Grader Benchmark

So sánh 2 models:
1. Base model: cross-encoder/nli-deberta-v3-base (hoặc MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
2. Fine-tuned v2: ./mdeberta_v3_medical_nli_v2

100 medical claims bao gồm:
- 55% Contradiction (chống chỉ định, liều quá liều, tương tác thuốc)
- 25% Entailment (suy luận đúng từ premise)
- 20% Neutral (không liên quan)
"""

import numpy as np
from sentence_transformers import CrossEncoder
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict

print("=" * 80)
print("🔬 NLI HALLUCINATION GRADER BENCHMARK")
print("=" * 80)

# ============================================================
# LOAD MODELS
# ============================================================
print("\n📦 Loading models...")

models = {}

# Base model (multilingual NLI)
try:
    models["Base (mDeBERTa-xnli)"] = CrossEncoder(
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        device='cuda'
    )
    print("   ✅ Base: mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
except Exception as e:
    print(f"   ❌ Base model failed: {e}")

# Fine-tuned v2
try:
    models["Fine-tuned v2"] = CrossEncoder(
        "./mdeberta_v3_medical_nli_v2",
        device='cuda'
    )
    print("   ✅ Fine-tuned: mdeberta_v3_medical_nli_v2")
except Exception as e:
    print(f"   ❌ Fine-tuned v2 failed: {e}")

if len(models) < 2:
    print("\n⚠️ Need both models to run comparison!")
    exit(1)

NLI_LABELS = ["entailment", "neutral", "contradiction"]

# ============================================================
# 100 MEDICAL TEST CASES
# ============================================================
test_cases = [
    # ============ CONTRADICTION CASES (55) ============
    # Medication Contraindications (20)
    {"doc": "Thuốc A không được chỉ định cho trẻ em dưới 12 tuổi.", "claim": "Thuốc A được chỉ định cho trẻ em.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Thuốc này không được dùng cho trẻ dưới 6 tuổi.", "claim": "Có thể cho trẻ 4 tuổi uống thuốc này.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Thuốc này không dành cho người cao huyết áp.", "claim": "Người cao huyết áp có thể dùng thuốc này.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không sử dụng cho bệnh nhân suy thận nặng.", "claim": "Bệnh nhân suy thận có thể dùng thuốc này.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Chống chỉ định cho người suy gan.", "claim": "Người suy gan có thể sử dụng thuốc này an toàn.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không dùng cho bệnh nhân glaucoma góc đóng.", "claim": "Bệnh nhân glaucoma có thể dùng thuốc này.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Thuốc chẹn beta không nên dùng cho bệnh nhân hen suyễn.", "claim": "Bệnh nhân hen có thể dùng thuốc chẹn beta.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Metformin chống chỉ định khi suy thận nặng.", "claim": "Người suy thận có thể dùng metformin.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không dùng NSAIDs cho bệnh nhân suy tim.", "claim": "Bệnh nhân suy tim có thể uống ibuprofen.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không nên dùng thuốc an thần mạnh cho người cao tuổi.", "claim": "Người già có thể dùng thuốc an thần mạnh.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Người dị ứng penicillin không được dùng amoxicillin.", "claim": "Người dị ứng penicillin có thể dùng amoxicillin.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "ACE inhibitors chống chỉ định khi có thai.", "claim": "Thai phụ có thể dùng thuốc ức chế ACE.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không dùng statin cho người đang mang thai.", "claim": "Phụ nữ mang thai có thể dùng statin.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Warfarin chống chỉ định trong thai kỳ.", "claim": "Thai phụ có thể dùng warfarin an toàn.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Isotretinoin tuyệt đối chống chỉ định khi mang thai.", "claim": "Phụ nữ mang thai có thể dùng isotretinoin.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không tiêm vaccine sống cho người suy giảm miễn dịch.", "claim": "Người suy giảm miễn dịch có thể tiêm vaccine sống.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không dùng aspirin cho bệnh nhân sốt xuất huyết.", "claim": "Bệnh nhân sốt xuất huyết có thể uống aspirin.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Ciprofloxacin không được dùng cho trẻ em.", "claim": "Trẻ em có thể dùng ciprofloxacin.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Thuốc này chống chỉ định khi đang cho con bú.", "claim": "Phụ nữ cho con bú có thể dùng thuốc này.", "expected": "contradiction", "category": "Contraindication"},
    {"doc": "Không dùng tetracycline cho trẻ dưới 8 tuổi.", "claim": "Trẻ 5 tuổi có thể dùng tetracycline.", "expected": "contradiction", "category": "Contraindication"},
    
    # Dosage/Limits (10)
    {"doc": "Không được dùng quá 4g paracetamol mỗi ngày.", "claim": "Có thể dùng 6g paracetamol mỗi ngày.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Liều tối đa ibuprofen là 2400mg mỗi ngày.", "claim": "Uống 3000mg ibuprofen mỗi ngày là an toàn.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Không nên bổ sung quá 4000 IU vitamin D mỗi ngày.", "claim": "Uống 10000 IU vitamin D hàng ngày là tốt.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Không dùng quá 45mg sắt nguyên tố mỗi ngày.", "claim": "Uống 100mg sắt mỗi ngày là an toàn.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Thai phụ không nên dùng quá 3000 IU vitamin A mỗi ngày.", "claim": "Thai phụ có thể uống 10000 IU vitamin A.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Không tiêu thụ quá 400mg caffeine mỗi ngày.", "claim": "Uống 800mg caffeine mỗi ngày là an toàn.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Acid folic dùng không quá 1000mcg mỗi ngày.", "claim": "Có thể dùng 5000mcg acid folic mỗi ngày.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Không bổ sung quá 40mg kẽm mỗi ngày.", "claim": "Uống 100mg kẽm mỗi ngày là tốt.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Liều aspirin tối đa là 4g mỗi ngày.", "claim": "Có thể uống 6g aspirin mỗi ngày.", "expected": "contradiction", "category": "Dosage"},
    {"doc": "Melatonin không nên dùng quá 10mg mỗi đêm.", "claim": "Uống 20mg melatonin mỗi đêm là an toàn.", "expected": "contradiction", "category": "Dosage"},
    
    # Drug Interactions (10)
    {"doc": "Không uống rượu khi đang dùng kháng sinh.", "claim": "Có thể uống rượu khi dùng kháng sinh.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Người dùng warfarin không nên ăn nhiều rau xanh giàu vitamin K.", "claim": "Bệnh nhân dùng warfarin có thể ăn nhiều rau cải.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Không dùng thực phẩm chứa tyramine khi uống thuốc MAOI.", "claim": "Có thể ăn phô mai khi đang dùng MAOI.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Không uống nước bưởi khi dùng thuốc statin.", "claim": "Nước bưởi an toàn khi dùng chung với statin.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Không dùng NSAIDs cùng methotrexate.", "claim": "Có thể dùng ibuprofen khi đang dùng methotrexate.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Không bổ sung kali khi dùng thuốc ức chế ACE.", "claim": "Người dùng ACE inhibitor có thể uống thêm kali.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Không dùng aspirin cùng với thuốc chống đông khác.", "claim": "Có thể uống aspirin khi đang dùng warfarin.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Tránh dùng sildenafil cùng nitrate.", "claim": "Có thể dùng Viagra khi đang dùng nitroglycerin.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Không phối hợp hai thuốc chống trầm cảm SSRI.", "claim": "Có thể dùng đồng thời fluoxetine và sertraline.", "expected": "contradiction", "category": "Interaction"},
    {"doc": "Clarithromycin không dùng cùng simvastatin.", "claim": "Có thể dùng clarithromycin với simvastatin.", "expected": "contradiction", "category": "Interaction"},
    
    # Medical Facts Contradiction (15)
    {"doc": "Bệnh tiểu đường không thể chữa khỏi hoàn toàn.", "claim": "Bệnh tiểu đường có thể chữa khỏi hoàn toàn.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Ung thư không phải là bệnh lây nhiễm.", "claim": "Ung thư có thể lây từ người này sang người khác.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Thuốc kháng sinh không có tác dụng với virus.", "claim": "Thuốc kháng sinh hiệu quả với virus.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Vắc xin không gây bệnh tự kỷ.", "claim": "Vắc xin có thể gây tự kỷ ở trẻ em.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Vắc xin COVID không gây vô sinh.", "claim": "Vắc xin COVID gây vô sinh.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Viêm gan B không lây qua đường ăn uống.", "claim": "Viêm gan B có thể lây qua đường ăn uống.", "expected": "contradiction", "category": "Facts"},
    {"doc": "HIV không lây qua tiếp xúc thông thường.", "claim": "HIV có thể lây qua bắt tay.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Bệnh nhân không có triệu chứng sốt.", "claim": "Bệnh nhân có triệu chứng sốt.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Rong kinh không phải là hiện tượng bình thường.", "claim": "Rong kinh là hiện tượng bình thường.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Hen suyễn không thể chữa khỏi nhưng có thể kiểm soát.", "claim": "Hen suyễn có thể chữa khỏi hoàn toàn.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Người không có triệu chứng vẫn có thể lây COVID.", "claim": "Chỉ người có triệu chứng mới lây COVID.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Thuốc có thể gây dị tật bẩm sinh cho thai nhi.", "claim": "Thuốc không ảnh hưởng đến thai nhi.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Không được sử dụng thuốc đã hết hạn.", "claim": "Thuốc hết hạn vẫn có thể sử dụng được.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Insulin không được để ở nhiệt độ cao.", "claim": "Insulin có thể để ngoài trời nắng.", "expected": "contradiction", "category": "Facts"},
    {"doc": "Trầm cảm là bệnh lý, không phải sự yếu đuối.", "claim": "Trầm cảm chỉ là sự yếu đuối tinh thần.", "expected": "contradiction", "category": "Facts"},
    
    # ============ ENTAILMENT CASES (25) ============
    {"doc": "Acid folic quan trọng trong thai kỳ vì giúp ngăn dị tật ống thần kinh.", "claim": "Acid folic cần thiết cho thai nhi.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Paracetamol là thuốc giảm đau và hạ sốt.", "claim": "Paracetamol có tác dụng giảm đau.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Bệnh tiểu đường cần kiểm soát đường huyết.", "claim": "Người tiểu đường phải theo dõi đường huyết.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Cao huyết áp làm tăng nguy cơ đột quỵ.", "claim": "Người cao huyết áp có nguy cơ đột quỵ cao hơn.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Rong kinh là tình trạng chảy máu kinh nguyệt kéo dài hơn 7 ngày.", "claim": "Rong kinh gây mất máu kéo dài.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Insulin giúp tế bào hấp thu glucose từ máu.", "claim": "Insulin điều hòa đường huyết.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Vitamin D giúp cơ thể hấp thu canxi.", "claim": "Vitamin D cần thiết cho xương.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Chất xơ giúp hệ tiêu hóa hoạt động tốt.", "claim": "Chất xơ tốt cho tiêu hóa.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Ngủ đủ giấc giúp phục hồi cơ thể và não bộ.", "claim": "Ngủ đủ tốt cho sức khỏe.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Tập thể dục đều đặn làm tim khỏe mạnh hơn.", "claim": "Tập thể dục tốt cho tim mạch.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Omega-3 có trong cá hồi tốt cho não bộ.", "claim": "Ăn cá hồi tốt cho não.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Hút thuốc lá gây ung thư phổi.", "claim": "Hút thuốc liên quan đến ung thư phổi.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Thuốc A được chỉ định cho người lớn.", "claim": "Thuốc A dùng được cho người trưởng thành.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Kháng sinh amoxicillin dùng để điều trị nhiễm khuẩn.", "claim": "Amoxicillin là thuốc kháng sinh.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Metformin là thuốc đầu tay điều trị tiểu đường type 2.", "claim": "Metformin dùng để điều trị tiểu đường.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Aspirin có tác dụng chống kết tập tiểu cầu.", "claim": "Aspirin giúp ngăn ngừa huyết khối.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Vắc xin COVID giúp giảm nguy cơ nhiễm bệnh nặng.", "claim": "Vắc xin COVID bảo vệ khỏi bệnh nặng.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Sắt cần thiết để tạo hemoglobin trong máu.", "claim": "Sắt quan trọng cho việc vận chuyển oxy.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Canxi cần thiết cho sự phát triển của xương.", "claim": "Canxi tốt cho xương.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Vitamin C giúp tăng cường hệ miễn dịch.", "claim": "Vitamin C tốt cho sức đề kháng.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Uống đủ nước giúp thận hoạt động tốt.", "claim": "Nước quan trọng cho chức năng thận.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Stress kéo dài làm tăng huyết áp.", "claim": "Căng thẳng ảnh hưởng đến huyết áp.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Thuốc lợi tiểu giúp giảm phù và hạ huyết áp.", "claim": "Thuốc lợi tiểu có thể dùng điều trị tăng huyết áp.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Đau thắt ngực là triệu chứng của bệnh mạch vành.", "claim": "Đau ngực có thể là dấu hiệu bệnh tim.", "expected": "entailment", "category": "Entailment"},
    {"doc": "Gan có chức năng lọc độc tố và sản xuất mật.", "claim": "Gan quan trọng cho việc giải độc cơ thể.", "expected": "entailment", "category": "Entailment"},
    
    # ============ NEUTRAL CASES (20) ============
    {"doc": "Acid folic quan trọng trong thai kỳ.", "claim": "Uống 2 lít nước mỗi ngày.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Paracetamol là thuốc giảm đau.", "claim": "Vitamin C có trong cam.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Bệnh tiểu đường cần kiểm soát đường huyết.", "claim": "Tập thể dục tốt cho tim mạch.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Viêm gan B là bệnh truyền nhiễm.", "claim": "Ung thư phổi liên quan đến hút thuốc.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Sốt là triệu chứng của nhiễm trùng.", "claim": "Đau lưng có thể do ngồi sai tư thế.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Canxi cần cho xương chắc khỏe.", "claim": "Insulin giúp điều hòa đường huyết.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Aspirin là thuốc giảm đau.", "claim": "Kháng sinh dùng để trị nhiễm khuẩn.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Tim bơm máu đi khắp cơ thể.", "claim": "Gan lọc độc tố trong máu.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Đau đầu có thể do căng thẳng.", "claim": "Đau bụng có thể do ăn uống không vệ sinh.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Vắc xin giúp phòng ngừa bệnh.", "claim": "Thuốc kháng sinh trị nhiễm khuẩn.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Cận thị cần đeo kính.", "claim": "Viêm tai giữa cần dùng kháng sinh.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Ngủ đủ 8 tiếng mỗi đêm.", "claim": "Ăn nhiều rau xanh tốt cho sức khỏe.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Thai kỳ kéo dài 40 tuần.", "claim": "Mãn kinh thường xảy ra ở tuổi 50.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Huyết áp bình thường dưới 120/80.", "claim": "Cholesterol cao làm tăng nguy cơ tim mạch.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Thận lọc máu và tạo nước tiểu.", "claim": "Phổi trao đổi oxy với môi trường.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Rong kinh là tình trạng kinh kéo dài.", "claim": "Tiểu đường type 1 cần tiêm insulin.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Thuốc A dùng buổi sáng.", "claim": "Thuốc B màu xanh.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Bác sĩ khám lúc 9 giờ sáng.", "claim": "Bệnh viện có phòng cấp cứu.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Uống thuốc sau bữa ăn.", "claim": "Tập thể dục 30 phút mỗi ngày.", "expected": "neutral", "category": "Neutral"},
    {"doc": "Vitamin D sản xuất khi tiếp xúc ánh nắng.", "claim": "Sắt có nhiều trong thịt đỏ.", "expected": "neutral", "category": "Neutral"},
]

print(f"\n📋 Total test cases: {len(test_cases)}")
print(f"   - Contradiction: {sum(1 for t in test_cases if t['expected'] == 'contradiction')}")
print(f"   - Entailment: {sum(1 for t in test_cases if t['expected'] == 'entailment')}")
print(f"   - Neutral: {sum(1 for t in test_cases if t['expected'] == 'neutral')}")

# ============================================================
# RUN BENCHMARK
# ============================================================
print("\n" + "=" * 80)
print("🏃 Running benchmark...")
print("=" * 80)

results = {}

for model_name, model in models.items():
    print(f"\n🔬 Testing: {model_name}")
    
    y_true = []
    y_pred = []
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})
    
    start_time = time.time()
    
    for i, test in enumerate(test_cases):
        scores = model.predict([(test['doc'], test['claim'])])[0]
        pred_idx = int(np.argmax(scores))
        predicted = NLI_LABELS[pred_idx]
        
        y_true.append(test['expected'])
        y_pred.append(predicted)
        
        is_correct = predicted == test['expected']
        category = test['category']
        category_results[category]["total"] += 1
        if is_correct:
            category_results[category]["correct"] += 1
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"   Processed {i + 1}/{len(test_cases)}")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    labels = ["entailment", "neutral", "contradiction"]
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    
    # Macro averages
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    results[model_name] = {
        "accuracy": accuracy,
        "precision": dict(zip(labels, precision)),
        "recall": dict(zip(labels, recall)),
        "f1": dict(zip(labels, f1)),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "category_results": dict(category_results),
        "time": elapsed,
        "y_true": y_true,
        "y_pred": y_pred
    }
    
    print(f"   ✅ Done in {elapsed:.2f}s | Accuracy: {accuracy*100:.1f}%")

# ============================================================
# RESULTS
# ============================================================
print("\n" + "=" * 80)
print("📊 BENCHMARK RESULTS")
print("=" * 80)

# Overall comparison
print("\n### Overall Accuracy")
print("-" * 50)
for name, res in results.items():
    acc = res['accuracy'] * 100
    f1 = res['macro_f1'] * 100
    print(f"{name:30} | Accuracy: {acc:5.1f}% | Macro-F1: {f1:5.1f}%")

# Per-class metrics
print("\n### Per-Class Performance")
print("-" * 70)
header = f"{'Model':25} | {'Class':15} | {'Precision':10} | {'Recall':10} | {'F1':10}"
print(header)
print("-" * 70)

for name, res in results.items():
    for cls in ["entailment", "neutral", "contradiction"]:
        p = res['precision'][cls] * 100
        r = res['recall'][cls] * 100
        f = res['f1'][cls] * 100
        print(f"{name:25} | {cls:15} | {p:9.1f}% | {r:9.1f}% | {f:9.1f}%")
    print("-" * 70)

# Category breakdown
print("\n### Performance by Category")
print("-" * 60)

categories = sorted(set(t['category'] for t in test_cases))
for cat in categories:
    print(f"\n{cat}:")
    for name, res in results.items():
        cat_res = res['category_results'].get(cat, {"correct": 0, "total": 0})
        correct = cat_res["correct"]
        total = cat_res["total"]
        acc = 100 * correct / total if total > 0 else 0
        print(f"   {name:30} | {correct}/{total} ({acc:.0f}%)")

# Confusion matrices
print("\n### Confusion Matrices")
print("-" * 50)

for name, res in results.items():
    print(f"\n{name}:")
    print("              Pred E   Pred N   Pred C")
    for i, true_label in enumerate(["Entail", "Neutral", "Contra"]):
        row = res['confusion_matrix'][i]
        print(f"  True {true_label:7} {row[0]:6}   {row[1]:6}   {row[2]:6}")

# Final summary
print("\n" + "=" * 80)
print("📊 FINAL SUMMARY")
print("=" * 80)

print("""
┌─────────────────────────────────┬────────────┬────────────┬────────────┐
│ Model                           │ Accuracy   │ Macro-F1   │ Time       │
├─────────────────────────────────┼────────────┼────────────┼────────────┤""")

for name, res in results.items():
    acc = res['accuracy'] * 100
    f1 = res['macro_f1'] * 100
    time_s = res['time']
    print(f"│ {name:31} │ {acc:8.1f}%  │ {f1:8.1f}%  │ {time_s:7.2f}s  │")

print("└─────────────────────────────────┴────────────┴────────────┴────────────┘")

# Winner
winner = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 WINNER: {winner[0]} (Accuracy: {winner[1]['accuracy']*100:.1f}%)")

# Critical category comparison (Contradiction is most important for hallucination detection)
print("\n### Critical: Contradiction Detection (Anti-Hallucination)")
print("-" * 60)
for name, res in results.items():
    p = res['precision']['contradiction'] * 100
    r = res['recall']['contradiction'] * 100
    f = res['f1']['contradiction'] * 100
    print(f"{name:30} | P={p:.0f}% R={r:.0f}% F1={f:.0f}%")
