"""
Streamlit App for Stage 5: User Profile Classifier (Refactored)

Призначення:
    Інтерактивний веб-додаток для класифікації профілів користувачів.
    Підтримка 6 попередньо підготовлених профілів з двома режимами роботи:
    - Повний режим (full): обробка зображень через ProfilePipeline
    - Швидкий режим (fast): завантаження готових результатів з JSON

Архітектура:
    Використовує модульний pipeline з model/stage_5/profile_pipeline.py:
    1. SigLIPClassifier - отримання raw scores
    2. ThresholdFilter - порогова фільтрація
    3. ProfileAggregator - агрегація профілю
    4. HierarchicalTagger - присвоєння тегів
    5. NoiseFilter - валідація тегів + recall корекція

Профілі (6 категорій):
    1. Catering → data/stage_5/cat/ + results/stage_5/cat/ [ПОВНИЙ РЕЖИМ]
    2. Marine_Activities → data/stage_5/mar/ + results/stage_5/mar/ [ПОВНИЙ РЕЖИМ]
    3. Cultural_Excursions → data/stage_5/cul/ + results/stage_5/cul/ [ШВИДКИЙ РЕЖИМ]
    4. Pet-Friendly Services → data/stage_5/pet/ + results/stage_5/pet/ [ШВИДКИЙ РЕЖИМ]
    5. Irrelevant → data/stage_5/irr/ + results/stage_5/irr/ [ШВИДКИЙ РЕЖИМ]
    6. Long_period → data/stage_5/base/ + results/stage_5/base/ [ШВИДКИЙ РЕЖИМ]

Вхідні дані:
    Для кожного профілю:
    - data/stage_5/{prefix}/*.jpg - зображення профілю
    - data/stage_5/{prefix}/image_timestamps.json - часові мітки
    - results/stage_5/{prefix}/profile_analysis.json - готові результати (для швидкого режиму)

Інтерфейс:
    1. Стартова сторінка:
       - 6 кнопок профілів (горизонтально, однакова ширина)
       
    2. Після вибору профілю:
       - Попередній перегляд зображень (перші 20, по 5 в ряду)
       - Кнопка "Аналізувати профіль"
       
    3. Після аналізу (вивід результатів):
       - Графік "Помісячна динаміка категорій"
       - Таблиця "Фінальний вектор інтересів" (відсотки)
       - Таблиця "Фінальні теги" (категорія + тег)

Режими роботи:
    ПОВНИЙ (full):
        - Завантаження зображень в пам'ять
        - Обробка через ProfilePipeline.analyze_profile()
        - Генерація raw scores, агрегація, тегізація, валідація
        - Використовується для: Catering, Marine_Activities
        
    ШВИДКИЙ (fast):
        - Завантаження готових результатів з profile_analysis.json
        - Миттєвий вивід результатів без обробки зображень
        - Використовується для: Cultural_Excursions, Pet-Friendly Services, Irrelevant, Long_period

Оптимізації:
    - st.session_state: зберігання завантажених зображень (уникнення перезавантаження)
    - st.cache_resource: кешування ProfilePipeline (завантаження моделі один раз)
    - Перемикання профілів: автоматичне очищення попередніх даних

Структура виводу:
    results = {
        'raw_counts': {"Catering": 4, ...},
        'factual_percentages': {"Catering": 40.0, ...},
        'tags': {"Catering": "User", ...},
        'validated_tags': {"Catering": "None", ...},
        'final_percentages': {"Catering": 0.0, "Irrelevant": 100.0, ...},
        'total_images': 10,
        'image_details': {
            'img1.jpg': {
                'raw_scores': {...},
                'filtered_category': 'Catering'
            }
        }
    }

Теги:
    - "Hobbyist": користувач-хоббіст (≥27.5% зображень категорії)
    - "User": користувач (≥12.8% зображень категорії)
    - "None": відсутній інтерес (<12.8% або не пройшла валідацію)

Модель:
    - SigLIP (google/siglip-base-patch16-224)
    - Ті самі промпти, що на Етапах 2-4
    - Кешування через @st.cache_resource
"""


import streamlit as st
from PIL import Image, ImageFile
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import sys

sys.path.append(str(Path(__file__).parent))
from model.stage_5.profile_pipeline import ProfilePipeline, BUSINESS_CATEGORIES

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROFILE_CONFIGS = {
    'Catering': {'data': 'data/stage_5/cat', 'results': 'results/stage_5/cat', 'mode': 'full'},
    'Marine': {'data': 'data/stage_5/mar', 'results': 'results/stage_5/mar', 'mode': 'full'},
    'Culture': {'data': 'data/stage_5/cul', 'results': 'results/stage_5/cul', 'mode': 'fast'},
    'Pets': {'data': 'data/stage_5/pet', 'results': 'results/stage_5/pet', 'mode': 'fast'},
    'Irrelevant': {'data': 'data/stage_5/irr', 'results': 'results/stage_5/irr', 'mode': 'fast'},
    'Timeline': {'data': 'data/stage_5/base', 'results': 'results/stage_5/base', 'mode': 'fast'}
}


@st.cache_resource
def load_pipeline():
    return ProfilePipeline()


def load_profile_data(profile_name):
    config = PROFILE_CONFIGS[profile_name]
    data_dir = Path(config['data'])
    
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png"))
    
    timestamps_path = data_dir / "image_timestamps.json"
    timestamps = {}
    if timestamps_path.exists():
        with open(timestamps_path, 'r') as f:
            timestamps = json.load(f)
    
    images = []
    filenames = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            filenames.append(img_path.name)
        except Exception as e:
            st.warning(f"Error loading {img_path.name}: {e}")
    
    return images, filenames, timestamps, config


def show_preview(images):
    st.subheader("Попередній перегляд зображень")
    preview_images = images[:20]
    
    for row_start in range(0, len(preview_images), 5):
        cols = st.columns(5)
        row_images = preview_images[row_start:row_start + 5]
        
        for col, image in zip(cols, row_images):
            with col:
                width, height = image.size
                if height > width:
                    image = image.rotate(90, expand=True)
                st.image(image, use_column_width=True)


def analyze_profile_full(images, timestamps):
    pipeline = load_pipeline()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    classifications = []
    image_details = {}
    filenames = list(timestamps.keys()) if timestamps else [f"image_{i}" for i in range(len(images))]
    
    for idx, (image, filename) in enumerate(zip(images, filenames)):
        status_text.text(f"Обробка зображення {idx + 1}/{len(images)}")
        
        scores = pipeline.classifier.process(image)
        category = pipeline.threshold_filter.process(scores)
        classifications.append(category)
        
        image_details[filename] = {
            'raw_scores': {k: round(v, 8) for k, v in scores.items()},
            'filtered_category': category
        }
        
        progress_bar.progress((idx + 1) / len(images))
    
    raw_counts = {cat: classifications.count(cat) for cat in ['Catering', 'Marine_Activities', 'Cultural_Excursions', 'Pet-Friendly Services', 'Irrelevant']}
    factual_percentages = pipeline.aggregator.process(classifications)
    tags = pipeline.tagger.process(factual_percentages)
    validated_tags, final_percentages = pipeline.noise_filter.process(raw_counts, tags, len(images))
    
    status_text.text("Аналіз завершено!")
    progress_bar.empty()
    status_text.empty()
    
    return {
        'raw_counts': raw_counts,
        'factual_percentages': factual_percentages,
        'tags': tags,
        'validated_tags': validated_tags,
        'final_percentages': final_percentages,
        'total_images': len(images),
        'image_details': image_details
    }


def analyze_profile_fast(profile_name):
    config = PROFILE_CONFIGS[profile_name]
    results_path = Path(config['results']) / "profile_analysis.json"
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def display_results(results, timestamps, profile_name):
    if 'monthly_dynamics' not in results and 'image_details' in results:
        monthly_counts = defaultdict(lambda: {cat: 0 for cat in BUSINESS_CATEGORIES + ['Irrelevant']})
        
        for filename, details in results['image_details'].items():
            if filename in timestamps:
                timestamp = datetime.fromisoformat(timestamps[filename])
                month_key = timestamp.strftime("%Y-%m")
                category = details['filtered_category']
                monthly_counts[month_key][category] += 1
    else:
        monthly_counts = results.get('monthly_dynamics', {})
    
    if monthly_counts and profile_name == 'Timeline':
        st.subheader("Помісячна динаміка категорій")
        
        sorted_months = sorted(monthly_counts.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = [0] * len(sorted_months)
        
        for category in BUSINESS_CATEGORIES:
            values = [monthly_counts[month][category] for month in sorted_months]
            ax.bar(sorted_months, values, bottom=bottom, label=category)
            bottom = [bottom[i] + values[i] for i in range(len(values))]
        
        values = [monthly_counts[month]['Irrelevant'] for month in sorted_months]
        ax.bar(sorted_months, values, bottom=bottom, label='Irrelevant', color='gray')
        
        ax.set_xlabel('Місяць')
        ax.set_ylabel('Кількість зображень')
        ax.set_title('Помісячна динаміка категорій')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    st.subheader("Фінальний вектор інтересів")
    df_result = pd.DataFrame([results['final_percentages']]).T.rename(columns={0: "Percentage (%)"})
    st.dataframe(df_result)
    
    if profile_name != 'Timeline':
        st.subheader("Фінальні теги")
        tags_data = []
        for category in BUSINESS_CATEGORIES:
            tags_data.append({
                'Категорія': category,
                'Тег': results['validated_tags'][category]
            })
        df_tags = pd.DataFrame(tags_data)
        st.dataframe(df_tags, hide_index=True)


def main():
    st.title("Класифікатор профілю користувача")
    
    if 'selected_profile' not in st.session_state:
        st.session_state.selected_profile = None
    if 'profile_data' not in st.session_state:
        st.session_state.profile_data = None
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    cols = st.columns([2, 2, 2, 2, 2, 2])
    profile_names = list(PROFILE_CONFIGS.keys())
    
    for idx, profile_name in enumerate(profile_names):
        with cols[idx]:
            if st.button(profile_name, key=f"btn_{profile_name}"):
                if st.session_state.selected_profile != profile_name:
                    st.session_state.selected_profile = profile_name
                    st.session_state.analyzed = False
                    
                    images, filenames, timestamps, config = load_profile_data(profile_name)
                    st.session_state.profile_data = {
                        'images': images,
                        'filenames': filenames,
                        'timestamps': timestamps,
                        'config': config
                    }
                    st.rerun()
    
    if st.session_state.selected_profile and st.session_state.profile_data:
        show_preview(st.session_state.profile_data['images'])
        
        if st.button("Аналізувати профіль"):
            config = st.session_state.profile_data['config']
            
            if config['mode'] == 'full':
                results = analyze_profile_full(
                    st.session_state.profile_data['images'],
                    st.session_state.profile_data['timestamps']
                )
            else:
                results = analyze_profile_fast(st.session_state.selected_profile)
            
            st.session_state.results = results
            st.session_state.analyzed = True
        
        if st.session_state.analyzed and 'results' in st.session_state:
            display_results(
                st.session_state.results,
                st.session_state.profile_data['timestamps'],
                st.session_state.selected_profile
            )


if __name__ == "__main__":
    main()