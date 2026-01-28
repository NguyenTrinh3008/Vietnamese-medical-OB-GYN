#!/usr/bin/env python3
"""
Comprehensive Medical NLI Test Suite

Extended test cases covering:
1. Medication contraindications & warnings
2. Dosage limits & numerical reasoning
3. Pregnancy & breastfeeding safety
4. Drug interactions
5. Disease conditions & symptoms
6. Treatment protocols
7. Vaccine-related
8. General medical negations
9. Entailment cases
10. Neutral/unrelated cases
"""

import numpy as np
from sentence_transformers import CrossEncoder
import time

print("=" * 100)
print("COMPREHENSIVE MEDICAL NLI TEST SUITE")
print("=" * 100)

# Models to compare
MODELS = {
    "Production": "./mdeberta_v3_production",
    "mDeBERTa v2 (Overfit)": "./mdeberta_v3_medical_nli_v2",
    "DeBERTa": "./vihealthnli_medical_finetuned",
}

print("\n🔧 Loading models...")
models = {}
for name, path in MODELS.items():
    try:
        models[name] = CrossEncoder(path, device='cuda')
        print(f"   ✅ {name}")
    except Exception as e:
        print(f"   ❌ {name}: {e}")

NLI_LABELS = ["entailment", "neutral", "contradiction"]

# ============================================================
# COMPREHENSIVE TEST CASES
# ============================================================

test_cases = [
    # =========================================================
    # 1. MEDICATION CONTRAINDICATIONS (15 cases)
    # =========================================================
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Pediatric Age Limit",
     "doc": "Thuốc A không được chỉ định cho trẻ em dưới 12 tuổi.",
     "claim": "Thuốc A được chỉ định cho trẻ em.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Pediatric Specific Age",
     "doc": "Thuốc này không được dùng cho trẻ dưới 6 tuổi.",
     "claim": "Có thể cho trẻ 4 tuổi uống thuốc này.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Hypertension Contraindication",
     "doc": "Thuốc này không dành cho người cao huyết áp.",
     "claim": "Người cao huyết áp có thể dùng thuốc này.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Kidney Disease",
     "doc": "Không sử dụng cho bệnh nhân suy thận nặng.",
     "claim": "Bệnh nhân suy thận có thể dùng thuốc này.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Liver Disease",
     "doc": "Chống chỉ định cho người suy gan.",
     "claim": "Người suy gan có thể sử dụng thuốc này an toàn.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Glaucoma Warning",
     "doc": "Không dùng cho bệnh nhân glaucoma góc đóng.",
     "claim": "Bệnh nhân glaucoma có thể dùng thuốc này.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Asthma Contraindication",
     "doc": "Thuốc chẹn beta không nên dùng cho bệnh nhân hen suyễn.",
     "claim": "Bệnh nhân hen có thể dùng thuốc chẹn beta.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Diabetes Metformin",
     "doc": "Metformin chống chỉ định khi suy thận nặng.",
     "claim": "Người suy thận có thể dùng metformin.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Heart Failure",
     "doc": "Không dùng NSAIDs cho bệnh nhân suy tim.",
     "claim": "Bệnh nhân suy tim có thể uống ibuprofen.",
     "expected": "contradiction"},
    
    {"category": "🚨 MEDICATION CONTRAINDICATION",
     "name": "Elderly Sedatives",
     "doc": "Không nên dùng thuốc an thần mạnh cho người cao tuổi.",
     "claim": "Người già có thể dùng thuốc an thần mạnh.",
     "expected": "contradiction"},
    
    # =========================================================
    # 2. DOSAGE LIMITS (10 cases)
    # =========================================================
    {"category": "💊 DOSAGE LIMITS",
     "name": "Paracetamol Overdose",
     "doc": "Không được dùng quá 4g paracetamol mỗi ngày.",
     "claim": "Có thể dùng 6g paracetamol mỗi ngày.",
     "expected": "contradiction"},
    
    {"category": "💊 DOSAGE LIMITS",
     "name": "Ibuprofen Max Dose",
     "doc": "Liều tối đa ibuprofen là 2400mg mỗi ngày.",
     "claim": "Uống 3000mg ibuprofen mỗi ngày là an toàn.",
     "expected": "contradiction"},
    
    {"category": "💊 DOSAGE LIMITS",
     "name": "Aspirin for Children",
     "doc": "Trẻ em dưới 16 tuổi không được dùng aspirin.",
     "claim": "Aspirin an toàn cho trẻ 10 tuổi.",
     "expected": "contradiction"},
    
    {"category": "💊 DOSAGE LIMITS",
     "name": "Vitamin D Upper Limit",
     "doc": "Không nên bổ sung quá 4000 IU vitamin D mỗi ngày.",
     "claim": "Uống 10000 IU vitamin D hàng ngày là tốt.",
     "expected": "contradiction"},
    
    {"category": "💊 DOSAGE LIMITS",
     "name": "Iron Supplement",
     "doc": "Không dùng quá 45mg sắt nguyên tố mỗi ngày.",
     "claim": "Uống 100mg sắt mỗi ngày là an toàn.",
     "expected": "contradiction"},
    
    {"category": "💊 DOSAGE LIMITS",
     "name": "Vitamin A Pregnancy",
     "doc": "Thai phụ không nên dùng quá 3000 IU vitamin A mỗi ngày.",
     "claim": "Thai phụ có thể uống 10000 IU vitamin A.",
     "expected": "contradiction"},
    
    {"category": "💊 DOSAGE LIMITS",
     "name": "Caffeine Limit",
     "doc": "Không nên tiêu thụ quá 400mg caffeine mỗi ngày.",
     "claim": "Uống 800mg caffeine mỗi ngày là an toàn.",
     "expected": "contradiction"},
    
    {"category": "💊 DOSAGE LIMITS",
     "name": "Zinc Limit",
     "doc": "Không bổ sung quá 40mg kẽm mỗi ngày.",
     "claim": "Uống 100mg kẽm mỗi ngày là tốt cho sức khỏe.",
     "expected": "contradiction"},
    
    # =========================================================
    # 3. PREGNANCY & BREASTFEEDING SAFETY (10 cases)
    # =========================================================
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "Pregnancy Warning General",
     "doc": "Không nên sử dụng thuốc này trong thai kỳ.",
     "claim": "Có thể sử dụng thuốc này trong thai kỳ.",
     "expected": "contradiction"},
    
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "First Trimester",
     "doc": "Thuốc này chống chỉ định trong 3 tháng đầu thai kỳ.",
     "claim": "Phụ nữ mang thai 2 tháng có thể dùng thuốc này.",
     "expected": "contradiction"},
    
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "Breastfeeding Warning",
     "doc": "Phụ nữ cho con bú không nên dùng thuốc này.",
     "claim": "Thuốc này an toàn khi cho con bú.",
     "expected": "contradiction"},
    
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "Teratogenic Effects",
     "doc": "Thuốc có thể gây dị tật bẩm sinh cho thai nhi.",
     "claim": "Thuốc không ảnh hưởng đến thai nhi.",
     "expected": "contradiction"},
    
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "Retinoid Pregnancy",
     "doc": "Retinoid tuyệt đối chống chỉ định khi mang thai.",
     "claim": "Thai phụ có thể dùng retinoid để trị mụn.",
     "expected": "contradiction"},
    
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "Alcohol Pregnancy",
     "doc": "Không có liều rượu nào an toàn trong thai kỳ.",
     "claim": "Uống một chút rượu khi mang thai không sao.",
     "expected": "contradiction"},
    
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "Third Trimester NSAIDs",
     "doc": "Không dùng NSAIDs trong 3 tháng cuối thai kỳ.",
     "claim": "Có thể uống ibuprofen khi thai 8 tháng.",
     "expected": "contradiction"},
    
    {"category": "🤰 PREGNANCY/BREASTFEEDING",
     "name": "Smoking Pregnancy",
     "doc": "Hút thuốc gây hại cho thai nhi.",
     "claim": "Hút thuốc không ảnh hưởng đến thai nhi.",
     "expected": "contradiction"},
    
    # =========================================================
    # 4. DRUG INTERACTIONS (10 cases)
    # =========================================================
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "Antibiotic + Alcohol",
     "doc": "Không uống rượu khi đang dùng kháng sinh.",
     "claim": "Có thể uống rượu khi dùng kháng sinh.",
     "expected": "contradiction"},
    
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "Warfarin + Vitamin K",
     "doc": "Người dùng warfarin không nên ăn nhiều rau xanh giàu vitamin K.",
     "claim": "Bệnh nhân dùng warfarin có thể ăn nhiều rau cải.",
     "expected": "contradiction"},
    
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "MAOI + Tyramine",
     "doc": "Không dùng thực phẩm chứa tyramine khi uống thuốc MAOI.",
     "claim": "Có thể ăn phô mai khi đang dùng MAOI.",
     "expected": "contradiction"},
    
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "Grapefruit Interaction",
     "doc": "Không uống nước bưởi khi dùng thuốc statin.",
     "claim": "Nước bưởi an toàn khi dùng chung với statin.",
     "expected": "contradiction"},
    
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "Methotrexate + NSAIDs",
     "doc": "Không dùng NSAIDs cùng methotrexate.",
     "claim": "Có thể dùng ibuprofen khi đang dùng methotrexate.",
     "expected": "contradiction"},
    
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "ACE Inhibitors + Potassium",
     "doc": "Không bổ sung kali khi dùng thuốc ức chế ACE.",
     "claim": "Người dùng ACE inhibitor có thể uống thêm kali.",
     "expected": "contradiction"},
    
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "Antidepressant Combination",
     "doc": "Không phối hợp hai thuốc chống trầm cảm khác nhóm.",
     "claim": "Có thể uống hai loại thuốc trầm cảm cùng lúc.",
     "expected": "contradiction"},
    
    {"category": "⚠️ DRUG INTERACTIONS",
     "name": "Blood Thinner Double",
     "doc": "Không dùng aspirin cùng với thuốc chống đông khác.",
     "claim": "Có thể uống aspirin khi đang dùng warfarin.",
     "expected": "contradiction"},
    
    # =========================================================
    # 5. DISEASE CONDITIONS & SYMPTOMS (10 cases)
    # =========================================================
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Fever Absence",
     "doc": "Bệnh nhân không có triệu chứng sốt.",
     "claim": "Bệnh nhân có triệu chứng sốt.",
     "expected": "contradiction"},
    
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Diabetes Cure Myth",
     "doc": "Bệnh tiểu đường không thể chữa khỏi hoàn toàn.",
     "claim": "Bệnh tiểu đường có thể chữa khỏi hoàn toàn.",
     "expected": "contradiction"},
    
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Cancer Not Contagious",
     "doc": "Ung thư không phải là bệnh lây nhiễm.",
     "claim": "Ung thư có thể lây từ người này sang người khác.",
     "expected": "contradiction"},
    
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Menorrhagia Abnormal",
     "doc": "Rong kinh không phải là hiện tượng bình thường.",
     "claim": "Rong kinh là hiện tượng bình thường.",
     "expected": "contradiction"},
    
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Headache Not Cancer",
     "doc": "Đau đầu thường không phải là dấu hiệu của ung thư não.",
     "claim": "Đau đầu thường là triệu chứng ung thư não.",
     "expected": "contradiction"},
    
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Allergy Not Immunity",
     "doc": "Dị ứng không có nghĩa là hệ miễn dịch yếu.",
     "claim": "Dị ứng chứng tỏ hệ miễn dịch yếu.",
     "expected": "contradiction"},
    
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Asthma Control",
     "doc": "Hen suyễn không thể chữa khỏi nhưng có thể kiểm soát được.",
     "claim": "Hen suyễn có thể chữa khỏi hoàn toàn.",
     "expected": "contradiction"},
    
    {"category": "🏥 SYMPTOMS/CONDITIONS",
     "name": "Depression Not Weakness",
     "doc": "Trầm cảm là bệnh lý, không phải sự yếu đuối.",
     "claim": "Trầm cảm chỉ là sự yếu đuối tinh thần.",
     "expected": "contradiction"},
    
    # =========================================================
    # 6. TREATMENT PROTOCOLS (8 cases)
    # =========================================================
    {"category": "💉 TREATMENT PROTOCOLS",
     "name": "Antibiotic Full Course",
     "doc": "Không được ngưng kháng sinh giữa chừng.",
     "claim": "Có thể ngừng kháng sinh khi hết triệu chứng.",
     "expected": "contradiction"},
    
    {"category": "💉 TREATMENT PROTOCOLS",
     "name": "Antibiotics vs Virus",
     "doc": "Thuốc kháng sinh không có tác dụng với virus.",
     "claim": "Thuốc kháng sinh hiệu quả với virus.",
     "expected": "contradiction"},
    
    {"category": "💉 TREATMENT PROTOCOLS",
     "name": "Self-Medication Warning",
     "doc": "Không nên tự ý mua thuốc kháng sinh.",
     "claim": "Có thể tự mua kháng sinh khi bị ho.",
     "expected": "contradiction"},
    
    {"category": "💉 TREATMENT PROTOCOLS",
     "name": "Insulin Storage",
     "doc": "Insulin không được để ở nhiệt độ cao.",
     "claim": "Insulin có thể để ngoài trời nắng.",
     "expected": "contradiction"},
    
    {"category": "💉 TREATMENT PROTOCOLS",
     "name": "Expired Medicine",
     "doc": "Không được sử dụng thuốc đã hết hạn.",
     "claim": "Thuốc hết hạn vẫn có thể sử dụng được.",
     "expected": "contradiction"},
    
    {"category": "💉 TREATMENT PROTOCOLS",
     "name": "Crushing Pills",
     "doc": "Không được nghiền viên thuốc phóng thích kéo dài.",
     "claim": "Có thể nghiền thuốc SR để uống dễ hơn.",
     "expected": "contradiction"},
    
    # =========================================================
    # 7. VACCINE-RELATED (8 cases)
    # =========================================================
    {"category": "💉 VACCINES",
     "name": "COVID Vaccine Fertility Myth",
     "doc": "Vắc xin COVID không gây vô sinh.",
     "claim": "Vắc xin COVID gây vô sinh.",
     "expected": "contradiction"},
    
    {"category": "💉 VACCINES",
     "name": "Live Vaccine Immunocompromised",
     "doc": "Không được tiêm vaccine sống cho người suy giảm miễn dịch.",
     "claim": "Người suy giảm miễn dịch có thể tiêm vaccine sống.",
     "expected": "contradiction"},
    
    {"category": "💉 VACCINES",
     "name": "Vaccine Autism Myth",
     "doc": "Vắc xin không gây bệnh tự kỷ.",
     "claim": "Vắc xin có thể gây tự kỷ ở trẻ em.",
     "expected": "contradiction"},
    
    {"category": "💉 VACCINES",
     "name": "Flu Vaccine Annual",
     "doc": "Cần tiêm vắc xin cúm hàng năm vì virus thay đổi.",
     "claim": "Tiêm vắc xin cúm một lần là đủ suốt đời.",
     "expected": "contradiction"},
    
    {"category": "💉 VACCINES",
     "name": "Vaccine Fever Safe",
     "doc": "Không tiêm vắc xin khi đang sốt cao.",
     "claim": "Có thể tiêm vắc xin khi đang sốt 39 độ.",
     "expected": "contradiction"},
    
    {"category": "💉 VACCINES",
     "name": "HPV Vaccine Age",
     "doc": "Vắc xin HPV hiệu quả nhất khi tiêm trước 26 tuổi.",
     "claim": "Vắc xin HPV chỉ dành cho người trên 30 tuổi.",
     "expected": "contradiction"},
    
    # =========================================================
    # 8. TRANSMISSION & INFECTION (7 cases)
    # =========================================================
    {"category": "🦠 TRANSMISSION",
     "name": "Hepatitis B Oral",
     "doc": "Viêm gan B không lây qua đường ăn uống.",
     "claim": "Viêm gan B có thể lây qua đường ăn uống.",
     "expected": "contradiction"},
    
    {"category": "🦠 TRANSMISSION",
     "name": "HIV Casual Contact",
     "doc": "HIV không lây qua tiếp xúc thông thường.",
     "claim": "HIV có thể lây qua bắt tay.",
     "expected": "contradiction"},
    
    {"category": "🦠 TRANSMISSION",
     "name": "TB Airborne",
     "doc": "Lao phổi lây qua đường hô hấp.",
     "claim": "Lao phổi không lây qua không khí.",
     "expected": "contradiction"},
    
    {"category": "🦠 TRANSMISSION",
     "name": "Malaria Mosquito",
     "doc": "Sốt rét chỉ lây qua muỗi đốt.",
     "claim": "Sốt rét có thể lây từ người sang người.",
     "expected": "contradiction"},
    
    {"category": "🦠 TRANSMISSION",
     "name": "COVID Asymptomatic",
     "doc": "Người không có triệu chứng vẫn có thể lây COVID.",
     "claim": "Chỉ người có triệu chứng mới lây COVID.",
     "expected": "contradiction"},
    
    # =========================================================
    # 9. ALLERGY WARNINGS (7 cases)  
    # =========================================================
    {"category": "🤧 ALLERGY",
     "name": "Penicillin Allergy",
     "doc": "Người dị ứng penicillin không được dùng amoxicillin.",
     "claim": "Người dị ứng penicillin có thể dùng amoxicillin.",
     "expected": "contradiction"},
    
    {"category": "🤧 ALLERGY",
     "name": "Sulfite Allergy",
     "doc": "Bệnh nhân hen dị ứng sulfite không dùng thuốc chứa sulfite.",
     "claim": "Người hen có thể dùng thuốc có sulfite.",
     "expected": "contradiction"},
    
    {"category": "🤧 ALLERGY",
     "name": "Iodine Contrast",
     "doc": "Người dị ứng iod không được chụp CT có cản quang.",
     "claim": "Người dị ứng iod có thể chụp CT cản quang.",
     "expected": "contradiction"},
    
    {"category": "🤧 ALLERGY",
     "name": "Latex Allergy",
     "doc": "Người dị ứng latex cần tránh găng tay cao su.",
     "claim": "Người dị ứng mủ cao su có thể dùng găng tay latex.",
     "expected": "contradiction"},
    
    {"category": "🤧 ALLERGY",
     "name": "Aspirin Allergy Triad",
     "doc": "Người dị ứng aspirin không được dùng NSAIDs.",
     "claim": "Người dị ứng aspirin có thể dùng ibuprofen.",
     "expected": "contradiction"},
    
    # =========================================================
    # 10. ENTAILMENT CASES (15 cases)
    # =========================================================
    {"category": "✅ ENTAILMENT",
     "name": "Exact Match",
     "doc": "Thuốc A được chỉ định cho người lớn.",
     "claim": "Thuốc A được chỉ định cho người lớn.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Folic Acid Pregnancy",
     "doc": "Acid folic quan trọng trong thai kỳ vì giúp ngăn dị tật ống thần kinh.",
     "claim": "Acid folic cần thiết cho thai nhi.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Paracetamol Function",
     "doc": "Paracetamol là thuốc giảm đau và hạ sốt.",
     "claim": "Paracetamol có tác dụng giảm đau.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Diabetes Management",
     "doc": "Bệnh tiểu đường cần kiểm soát đường huyết.",
     "claim": "Người tiểu đường phải theo dõi đường huyết.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Hypertension Risk",
     "doc": "Cao huyết áp làm tăng nguy cơ đột quỵ.",
     "claim": "Người cao huyết áp có nguy cơ đột quỵ cao hơn.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Menorrhagia Definition",
     "doc": "Rong kinh là tình trạng chảy máu kinh nguyệt kéo dài hơn 7 ngày.",
     "claim": "Rong kinh gây mất máu kéo dài.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Insulin Function",
     "doc": "Insulin giúp tế bào hấp thu glucose từ máu.",
     "claim": "Insulin điều hòa đường huyết.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Vitamin D Calcium",
     "doc": "Vitamin D giúp cơ thể hấp thu canxi.",
     "claim": "Vitamin D cần thiết cho xương.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Fiber Digestion",
     "doc": "Chất xơ giúp hệ tiêu hóa hoạt động tốt.",
     "claim": "Chất xơ tốt cho tiêu hóa.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Sleep Importance",
     "doc": "Ngủ đủ giấc giúp phục hồi cơ thể và não bộ.",
     "claim": "Ngủ đủ tốt cho sức khỏe.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Exercise Heart",
     "doc": "Tập thể dục đều đặn làm tim khỏe mạnh hơn.",
     "claim": "Tập thể dục tốt cho tim mạch.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Omega3 Brain",
     "doc": "Omega-3 có trong cá hồi tốt cho não bộ.",
     "claim": "Ăn cá hồi tốt cho não.",
     "expected": "entailment"},
    
    {"category": "✅ ENTAILMENT",
     "name": "Smoking Lung",
     "doc": "Hút thuốc lá gây ung thư phổi.",
     "claim": "Hút thuốc liên quan đến ung thư phổi.",
     "expected": "entailment"},
    
    # =========================================================
    # 11. NEUTRAL CASES (15 cases)
    # =========================================================
    {"category": "⚪ NEUTRAL",
     "name": "Water vs Folic Acid",
     "doc": "Acid folic quan trọng trong thai kỳ.",
     "claim": "Uống 2 lít nước mỗi ngày.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Paracetamol vs Vitamin C",
     "doc": "Paracetamol là thuốc giảm đau.",
     "claim": "Vitamin C có trong cam.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Diabetes vs Exercise",
     "doc": "Bệnh tiểu đường cần kiểm soát đường huyết.",
     "claim": "Tập thể dục tốt cho tim mạch.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Different Medical Topics",
     "doc": "Viêm gan B là bệnh truyền nhiễm.",
     "claim": "Ung thư phổi liên quan đến hút thuốc.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Fever vs Back Pain",
     "doc": "Sốt là triệu chứng của nhiễm trùng.",
     "claim": "Đau lưng có thể do ngồi sai tư thế.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Calcium vs Insulin",
     "doc": "Canxi cần cho xương chắc khỏe.",
     "claim": "Insulin giúp điều hòa đường huyết.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Aspirin vs Antibiotic",
     "doc": "Aspirin là thuốc giảm đau.",
     "claim": "Kháng sinh dùng để trị nhiễm khuẩn.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Heart vs Liver",
     "doc": "Tim bơm máu đi khắp cơ thể.",
     "claim": "Gan lọc độc tố trong máu.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Headache vs Stomachache",
     "doc": "Đau đầu có thể do căng thẳng.",
     "claim": "Đau bụng có thể do ăn uống không hợp vệ sinh.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Vaccine vs Medicine",
     "doc": "Vắc xin giúp phòng ngừa bệnh.",
     "claim": "Thuốc kháng sinh trị nhiễm khuẩn.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Eyes vs Ears",
     "doc": "Cận thị cần đeo kính.",
     "claim": "Viêm tai giữa cần dùng kháng sinh.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Sleep vs Diet",
     "doc": "Ngủ đủ 8 tiếng mỗi đêm.",
     "claim": "Ăn nhiều rau xanh tốt cho sức khỏe.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Pregnancy vs Menopause",
     "doc": "Thai kỳ kéo dài 40 tuần.",
     "claim": "Mãn kinh thường xảy ra ở tuổi 50.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Blood Pressure vs Cholesterol",
     "doc": "Huyết áp bình thường dưới 120/80.",
     "claim": "Cholesterol cao làm tăng nguy cơ tim mạch.",
     "expected": "neutral"},
    
    {"category": "⚪ NEUTRAL",
     "name": "Kidney vs Lung",
     "doc": "Thận lọc máu và tạo nước tiểu.",
     "claim": "Phổi trao đổi oxy với môi trường.",
     "expected": "neutral"},
]

print(f"\n📋 Total test cases: {len(test_cases)}")
print("=" * 100)

# Run tests
results = {name: {"passed": 0, "failed": 0, "by_category": {}} for name in models.keys()}

for i, test in enumerate(test_cases, 1):
    category = test["category"]
    
    print(f"\n[{i}/{len(test_cases)}] {category} - {test['name']}")
    print(f"   Doc: {test['doc'][:60]}...")
    print(f"   Claim: {test['claim']}")
    print(f"   Expected: {test['expected'].upper()}")
    print("-" * 80)
    
    for model_name, model in models.items():
        scores = model.predict([(test['doc'], test['claim'])])[0]
        pred_idx = np.argmax(scores)
        predicted = NLI_LABELS[pred_idx]
        confidence = scores[pred_idx]
        
        is_correct = predicted == test['expected']
        status = "✅" if is_correct else "❌"
        
        if is_correct:
            results[model_name]["passed"] += 1
        else:
            results[model_name]["failed"] += 1
        
        # Track by category
        if category not in results[model_name]["by_category"]:
            results[model_name]["by_category"][category] = {"passed": 0, "total": 0}
        results[model_name]["by_category"][category]["total"] += 1
        if is_correct:
            results[model_name]["by_category"][category]["passed"] += 1
        
        print(f"   {model_name}: {predicted.upper():12} conf={confidence:+.2f} {status}")

# Summary
print("\n" + "=" * 100)
print("OVERALL RESULTS")
print("=" * 100)

for model_name in models.keys():
    passed = results[model_name]["passed"]
    total = passed + results[model_name]["failed"]
    accuracy = 100 * passed / total
    print(f"\n📊 {model_name}: {passed}/{total} ({accuracy:.1f}%)")

# Category breakdown
print("\n" + "=" * 100)
print("RESULTS BY CATEGORY")
print("=" * 100)

categories = sorted(set(test["category"] for test in test_cases))

# Header
header = f"{'Category':<35}"
for model_name in models.keys():
    header += f" | {model_name[:15]:>15}"
print(header)
print("-" * len(header))

for category in categories:
    row = f"{category:<35}"
    for model_name in models.keys():
        cat_result = results[model_name]["by_category"].get(category, {"passed": 0, "total": 0})
        passed = cat_result["passed"]
        total = cat_result["total"]
        row += f" | {passed}/{total}".rjust(16)
    print(row)

# Final summary table
print("\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)

print(f"""
┌─────────────────────────────────┬──────────┬───────────────┐
│ Model                           │ Passed   │ Accuracy      │
├─────────────────────────────────┼──────────┼───────────────┤""")
for model_name in models.keys():
    passed = results[model_name]["passed"]
    total = passed + results[model_name]["failed"]
    accuracy = 100 * passed / total
    print(f"│ {model_name:<31} │ {passed}/{total}".ljust(43) + f"│ {accuracy:>10.1f}%   │")
print("└─────────────────────────────────┴──────────┴───────────────┘")

# Find failures
print("\n" + "=" * 100)
print("FAILED CASES ANALYSIS")
print("=" * 100)

for model_name, model in models.items():
    failures = []
    for test in test_cases:
        scores = model.predict([(test['doc'], test['claim'])])[0]
        pred_idx = np.argmax(scores)
        predicted = NLI_LABELS[pred_idx]
        if predicted != test['expected']:
            failures.append({
                "category": test["category"],
                "name": test["name"],
                "expected": test["expected"],
                "predicted": predicted
            })
    
    if failures:
        print(f"\n❌ {model_name} failed {len(failures)} cases:")
        for f in failures:
            print(f"   - [{f['category']}] {f['name']}: Expected {f['expected'].upper()}, got {f['predicted'].upper()}")
    else:
        print(f"\n✅ {model_name}: All tests passed!")
