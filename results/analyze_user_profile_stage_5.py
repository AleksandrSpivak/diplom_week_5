"""
User Profile Analyzer for Stage 5: CLI Version (Refactored)

Призначення:
    Аналіз профілю користувача з локальної директорії зображень.
    Генерація фінального 100% нормалізованого вектору інтересів.
    Побудова помісячної динаміки категорій.

Процес аналізу:
    Використовує модульний pipeline з model/stage_5/profile_pipeline.py:
    1. SigLIPClassifier - отримання raw scores
    2. ThresholdFilter - порогова фільтрація
    3. ProfileAggregator - агрегація профілю
    4. HierarchicalTagger - присвоєння тегів
    5. NoiseFilter - валідація тегів + recall корекція

Вхідні дані:
    - data/stage_5/images/*.jpg - зображення профілю користувача
    - data/stage_5/images/image_timestamps.json - часові мітки зображень

Вихідні дані:
    - results/stage_5/profile_analysis.json - фінальний вектор інтересів + теги
    - results/stage_5/monthly_dynamics.png - помісячна динаміка категорій
"""

from PIL import Image, ImageFile
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import sys

# Додати шлях до model/stage_5
sys.path.append(str(Path(__file__).parent.parent))

from model.stage_5.profile_pipeline import ProfilePipeline, BUSINESS_CATEGORIES

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    print("=" * 60)
    print("STAGE 5: User Profile Analysis")
    print("=" * 60)

    # Шляхи
    images_dir = Path("data/stage_5/irr")
    timestamps_path = images_dir / "image_timestamps.json"
    output_dir = Path("results/stage_5/irr")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "profile_analysis.json"
    plot_path = output_dir / "monthly_dynamics.png"

    # Завантаження часових міток
    with open(timestamps_path, "r") as f:
        timestamps = json.load(f)

    # Отримання списку зображень
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    
    if not image_files:
        print(f"⚠ No images found in {images_dir}")
        return

    total_images = len(image_files)
    print(f"\nTotal images to analyze: {total_images}")

    # Ініціалізація pipeline
    print("\nInitializing ProfilePipeline...")
    pipeline = ProfilePipeline()

    # Завантаження зображень
    print("\nLoading images...")
    images = []
    image_filenames = []
    for image_path in tqdm(image_files, desc="Loading images"):
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            image_filenames.append(image_path.name)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")

    # Аналіз профілю через pipeline
    print("\nAnalyzing profile...")
    results = pipeline.analyze_profile(images, timestamps)

    # Помісячний підрахунок (з image_details)
    print("\nCalculating monthly dynamics...")
    monthly_counts = defaultdict(lambda: {
        'Catering': 0,
        'Marine_Activities': 0,
        'Cultural_Excursions': 0,
        'Pet-Friendly Services': 0,
        'Irrelevant': 0
    })

    for filename, details in results['image_details'].items():
        if filename in timestamps:
            timestamp = datetime.fromisoformat(timestamps[filename])
            month_key = timestamp.strftime("%Y-%m")
            category = details['filtered_category']
            monthly_counts[month_key][category] += 1

    # Вивід проміжних результатів
    print(f"\n{'='*60}")
    print("1. Початковий розподіл сигналів (Raw Counts)")
    print(f"{'='*60}")
    for category, count in results['raw_counts'].items():
        print(f"  {category}: {count}")

    # Вивід тегів (до валідації)
    print(f"\n{'='*60}")
    print("2. Теги бізнес-категорій (на фактичних даних)")
    print(f"{'='*60}")
    for category, tag in results['tags'].items():
        print(f"  {category}: {tag}")

    # Вивід валідованих тегів
    print(f"\n{'='*60}")
    print("3. Валідовані теги (після перевірки мінімальних порогів)")
    print(f"{'='*60}")
    for category, tag in results['validated_tags'].items():
        print(f"  {category}: {tag}")

    # Вивід фінального результату
    print(f"\n{'='*60}")
    print("4. Фінальний вектор інтересів")
    print("   (після фільтрації та валідації)")
    print(f"{'='*60}")
    for category, percentage in results['final_percentages'].items():
        print(f"  {category}: {percentage}%")

    # Збереження результатів
    results_to_save = {
        "total_images": results['total_images'],
        "raw_counts": results['raw_counts'],
        "factual_percentages": results['factual_percentages'],
        "tags": results['tags'],
        "validated_tags": results['validated_tags'],
        "final_percentages": results['final_percentages'],
        "monthly_dynamics": dict(monthly_counts)
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)

    # Побудова помісячної діаграми
    print(f"\n{'='*60}")
    print("5. Побудова помісячної динаміки")
    print(f"{'='*60}")
    
    sorted_months = sorted(monthly_counts.keys())
    
    if sorted_months:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Підготовка даних для stacked bar chart
        bottom = [0] * len(sorted_months)
        
        # Спочатку бізнес-категорії
        for category in BUSINESS_CATEGORIES:
            values = [monthly_counts[month][category] for month in sorted_months]
            ax.bar(sorted_months, values, bottom=bottom, label=category)
            bottom = [bottom[i] + values[i] for i in range(len(values))]
        
        # Потім Irrelevant сірим кольором
        values = [monthly_counts[month]['Irrelevant'] for month in sorted_months]
        ax.bar(sorted_months, values, bottom=bottom, label='Irrelevant', color='gray')
        
        ax.set_xlabel('Місяць')
        ax.set_ylabel('Кількість зображень')
        ax.set_title('Помісячна динаміка категорій (до фільтрації)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(plot_path, dpi=300)
        print(f"Діаграма збережена: {plot_path}")
    else:
        print("⚠ Немає даних для побудови графіка (відсутні timestamps)")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()