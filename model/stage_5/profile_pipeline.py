"""
Profile Analysis Pipeline for Stage 5: Modular Architecture

Призначення:
    Модульна система аналізу профілю користувача для визначення бізнес-інтересів
    на основі зображень. П'ятиетапний pipeline з послідовною обробкою від raw scores
    до фінального нормалізованого вектору з тегами.

Архітектура (5 модулів):
    1. SigLIPClassifier - Zero-Shot класифікація зображень
       Вхід: PIL Image
       Вихід: {"Catering": 0.456, "Marine_Activities": 0.012, ...} (raw scores)
       
    2. ThresholdFilter - Порогова фільтрація на рівні зображення
       Вхід: raw scores від SigLIPClassifier
       Вихід: назва категорії (з фільтром FILTER_THRESHOLD = 0.000010)
       Логіка: Якщо бізнес-категорія має score < 0.000010 → "Irrelevant"
       
    3. ProfileAggregator - Агрегація результатів профілю
       Вхід: список категорій для всіх зображень профілю
       Вихід: {"Catering": 25.5, "Marine_Activities": 0.0, ...} (фактичні %)
       Логіка: Підрахунок частки кожної категорії в профілі
       
    4. HierarchicalTagger - Присвоєння тегів на основі порогів
       Вхід: фактичні % після агрегації ("брудні" дані)
       Вихід: {"Catering": "Hobbyist", "Marine_Activities": "None", ...}
       Пороги: Hobbyist ≥ 27.5%, User ≥ 12.8%, інакше None
       
    5. NoiseFilter - Фільтрація шуму, валідація тегів та recall корекція
    Вхід: raw_counts {"Catering": 4, ...}, tags {"Catering": "Hobbyist", ...}, total_images
    Вихід: 
        - validated_tags: {"Catering": "None", "Marine_Activities": "None", ...} (теги після валідації)
        - final_percentages: {"Catering": 0.0, "Irrelevant": 100.0, ...} (фінальні %, 100% сума)
    Логіка: 
        - Перевірка мінімальних порогів (DETECTION_THRESHOLDS)
        - Якщо raw_counts[category] < DETECTION_THRESHOLDS для категорії → обнулити counts І tags[category] = "None"
        - Recall корекція (RECALL_FACTORS) з округленням
        - "Нульова сума" (delta перерозподіл до Irrelevant)
        - 100% нормалізація

Константи з дослідження:
    FILTER_THRESHOLD = 0.000010 (Етап 3)
    DETECTION_THRESHOLDS = {Catering:5, Marine:3, Cultural:4, Pet:2} (Етап 4)
    RECALL_FACTORS = {Catering:0.872, Marine:0.758, Cultural:0.887, Pet:0.772} (Етап 4)
    TAGGER_THRESHOLDS = {Hobbyist:0.275, User:0.128}

ProfilePipeline (Orchestrator):
    Метод analyze_profile(images: List[Image], timestamps: Dict[str, str] = None) повертає:
    {
        'raw_counts': {"Catering": 4, "Marine_Activities": 0, ...},
        'factual_percentages': {"Catering": 40.0, ...} (до фільтрації),
        'tags': {"Catering": "User", "Marine_Activities": "None", ...},
        'validated_tags': {"Catering": "None", ...} (після валідації),
        'final_percentages': {"Catering": 0.0, "Irrelevant": 100.0, ...} (після всіх корекцій),
        'total_images': 10,
        'image_details': {
            'img1.jpg': {
                'raw_scores': {"Catering": 0.00456789, "Marine_Activities": 0.00012345, ...},
                'filtered_category': 'Catering'
            },
            'img2.jpg': {
                'raw_scores': {...},
                'filtered_category': 'Irrelevant'
            }
        }
    }
    
    Параметри:
        images: список PIL Images
        timestamps: (опціонально) словник {"filename.jpg": "2025-07-15T10:00:00", ...}
                   - якщо передано: використовуються реальні імена файлів як ключі
                   - якщо None: генеруються "image_0", "image_1", ... як ключі

Використання:
    from model.stage_5.profile_pipeline import ProfilePipeline
    
    # З timestamps
    pipeline = ProfilePipeline()
    results = pipeline.analyze_profile(images, timestamps)
    
    # Без timestamps
    results = pipeline.analyze_profile(images)
"""


import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


FILTER_THRESHOLD = 0.000010

DETECTION_THRESHOLDS = {
    'Catering': 5,
    'Marine_Activities': 3,
    'Cultural_Excursions': 4,
    'Pet-Friendly Services': 2
}

RECALL_FACTORS = {
    'Catering': 0.872,
    'Marine_Activities': 0.758,
    'Cultural_Excursions': 0.887,
    'Pet-Friendly Services': 0.772
}

TAGGER_THRESHOLDS = {
    'Hobbyist': 0.275,
    'User': 0.128
}

CLASS_PROMPTS = {
    "Catering": "a photo of food or drinks, restaurant, dining",
    "Marine_Activities": "a photo of watercraft, boat, jet ski, surfing, beach activities",
    "Cultural_Excursions": "a photo of cultural landmark, monument, museum, castle, sculpture",
    "Pet-Friendly Services": "a photo of domestic pet, dog, cat, puppy, kitten",
    "Irrelevant": "a photo unrelated to business services",
}

BUSINESS_CATEGORIES = ['Catering', 'Marine_Activities', 'Cultural_Excursions', 'Pet-Friendly Services']


class SigLIPClassifier:
    
    def __init__(self, model_name="google/siglip-base-patch16-224", model_path="weights/siglip"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_dir = Path(model_path)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, cache_dir=str(model_dir)
        )
        self.model.to(self.device)
    
    def process(self, image: Image.Image) -> Dict[str, float]:
        scores = {}

        for class_name, prompt in CLASS_PROMPTS.items():
            inputs = self.processor(
                text=[prompt], images=image, padding="max_length", return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                prob = torch.sigmoid(logits_per_image)

            scores[class_name] = prob[0][0].cpu().item()

        return scores


class ThresholdFilter:
    
    def __init__(self, threshold: float = FILTER_THRESHOLD):
        self.threshold = threshold
    
    def process(self, scores: Dict[str, float]) -> str:
        winner_class = max(scores, key=scores.get)
        winner_score = scores[winner_class]
        
        if winner_class in BUSINESS_CATEGORIES and winner_score < self.threshold:
            return 'Irrelevant'
        
        return winner_class


class ProfileAggregator:
    
    def process(self, classifications: List[str]) -> Dict[str, float]:
        total = len(classifications)
        counts = {
            'Catering': 0,
            'Marine_Activities': 0,
            'Cultural_Excursions': 0,
            'Pet-Friendly Services': 0,
            'Irrelevant': 0
        }
        
        for category in classifications:
            counts[category] += 1
        
        percentages = {
            category: (count / total) * 100
            for category, count in counts.items()
        }
        
        return percentages


class HierarchicalTagger:
    
    def __init__(self, thresholds: Dict[str, float] = TAGGER_THRESHOLDS):
        self.thresholds = thresholds
    
    def process(self, percentages: Dict[str, float]) -> Dict[str, str]:
        tags = {}
        
        for category in BUSINESS_CATEGORIES:
            percentage = percentages.get(category, 0.0) / 100
            
            if percentage >= self.thresholds['Hobbyist']:
                tags[category] = 'Hobbyist'
            elif percentage >= self.thresholds['User']:
                tags[category] = 'User'
            else:
                tags[category] = 'None'
        
        return tags


class NoiseFilter:
    
    def __init__(self, 
                 detection_thresholds: Dict[str, int] = DETECTION_THRESHOLDS,
                 recall_factors: Dict[str, float] = RECALL_FACTORS):
        self.detection_thresholds = detection_thresholds
        self.recall_factors = recall_factors
    
    def process(self, 
                raw_counts: Dict[str, int],
                tags: Dict[str, str],
                total_images: int) -> tuple[Dict[str, str], Dict[str, float]]:
        final_counts = raw_counts.copy()
        validated_tags = tags.copy()
        filtered_out_count = 0
        
        for category in BUSINESS_CATEGORIES:
            if raw_counts[category] < self.detection_thresholds[category]:
                filtered_out_count += final_counts[category]
                final_counts[category] = 0
                validated_tags[category] = "None"
        
        final_counts['Irrelevant'] += filtered_out_count
        
        corrected_counts = {}
        delta_sum = 0
        
        for category in BUSINESS_CATEGORIES:
            if category in self.recall_factors and final_counts[category] > 0:
                corrected_float = final_counts[category] / self.recall_factors[category]
                corrected_int = round(corrected_float)
                delta = corrected_int - final_counts[category]
                delta_sum += delta
                corrected_counts[category] = corrected_int
            else:
                corrected_counts[category] = final_counts[category]
        
        corrected_counts['Irrelevant'] = final_counts['Irrelevant'] - delta_sum
        
        final_vector_normalized = {}
        for category in raw_counts.keys():
            percentage = (corrected_counts[category] / total_images) * 100
            final_vector_normalized[category] = round(percentage, 2)
        
        return validated_tags, final_vector_normalized


class ProfilePipeline:
    
    def __init__(self):
        self.classifier = SigLIPClassifier()
        self.threshold_filter = ThresholdFilter()
        self.aggregator = ProfileAggregator()
        self.tagger = HierarchicalTagger()
        self.noise_filter = NoiseFilter()
    
    def analyze_profile(self, images: List[Image.Image], timestamps: Dict[str, str] = None) -> dict:
        classifications = []
        image_details = {}
        
        filenames = list(timestamps.keys()) if timestamps else [f"image_{i}" for i in range(len(images))]
        
        for image, filename in tqdm(zip(images, filenames), total=len(images), desc="Analyzing images"):
            scores = self.classifier.process(image)
            category = self.threshold_filter.process(scores)
            classifications.append(category)
            
            image_details[filename] = {
                'raw_scores': {k: round(v, 8) for k, v in scores.items()},
                'filtered_category': category
            }
        
        raw_counts = {cat: classifications.count(cat) for cat in CLASS_PROMPTS.keys()}
        
        factual_percentages = self.aggregator.process(classifications)
        
        tags = self.tagger.process(factual_percentages)
        
        validated_tags, final_percentages = self.noise_filter.process(raw_counts, tags, len(images))
        
        return {
            'raw_counts': raw_counts,
            'factual_percentages': factual_percentages,
            'tags': tags,
            'validated_tags': validated_tags,
            'final_percentages': final_percentages,
            'total_images': len(images),
            'image_details': image_details
        }