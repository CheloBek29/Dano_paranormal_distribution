import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

df = pd.read_csv('final_dtp.csv')
with open('regions_federal_districts_89.json', 'r', encoding='utf-8') as f:
    region_map = json.load(f)


if 'region_name' in df.columns:
    region_col = 'region_name'
elif 'REGION' in df.columns:
    region_col = 'REGION'

# Используем .map(). Если региона нет в словаре, будет NaN
df['district'] = df[region_col].map(region_map)

# Удаляем строки, где округ не определился
df_model = df.dropna(subset=['district']).copy()
print(f"Размер данных после удаления NaN в district: {df_model.shape}")


# Проверяем наличие всех необходимых переменных
required_vars = ['district', 'n_VEHICLES', 'vehicle_failure', 'female_driver', 'guilty_exp_avg',
       'no_seatbelt_injury', 'n_guilty', 'road_defects_cat', 'road_surface_cat',
       'impaired_driving', 'wrong_way', 'lighting_cat', 'SEASON']

# Создаем df только с нужными переменными (без целевой, так как ее пока нет)
df = df_model[required_vars].copy()

quant_vars = ['n_VEHICLES', 'n_guilty', 'guilty_exp_avg']
binary_vars = ['vehicle_failure', 'female_driver', 'no_seatbelt_injury',
               'impaired_driving', 'wrong_way']
cat_vars = ['district', 'road_defects_cat',
            'road_surface_cat', 'lighting_cat', 'SEASON']
df_target = df_model['severity'].copy()

base_categories = {
    'lighting_cat': 0,
    'road_surface_cat': 7,  # Сухое покрытие
    'road_defects_cat': 5,  # Недостатки содержания (или Нет дефектов)
    'SEASON': 3,  # Сравниваем Зиму/Осень с Летом
    'district': 'Центральный',
}



print("\n" + "=" * 80)
print("ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ В ДАММИ")
print("=" * 80)

# Создаем копию данных для преобразования
df_encoded = df.copy()


# Функция для создания дамми с заданной базовой категорией
def create_dummies_with_base(df, column, base_category=None):

    original_values = df[column].unique()

    # Получаем все уникальные категории
    all_categories = sorted(df[column].astype(str).unique())

    # Если базовая категория указана
    if base_category is not None:
        base_category = str(base_category)  # Преобразуем в строку

        if base_category in all_categories:
            # Упорядочиваем: сначала базовая, потом остальные
            ordered_categories = [base_category] + [c for c in all_categories if c != base_category]

            # Преобразуем в Categorical тип
            df[column] = pd.Categorical(
                df[column].astype(str),  # Преобразуем в строку
                categories=ordered_categories,
                ordered=False
            )
        #     print(f"    Установлена базовая категория: '{base_category}'")
        # else:
        #     print(f"    ВНИМАНИЕ: Базовая категория '{base_category}' не найдена!")
        #     print(f"    Доступные значения: {all_categories}")
        #     print(f"    Использую первую по алфавиту: '{all_categories[0]}'")

    # Создаем дамми-переменные
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True, dtype=int)

    # УДАЛЯЕМ исходную колонку
    df = df.drop(column, axis=1)

    # Объединяем с дамми
    df = pd.concat([df, dummies], axis=1)

    return df


for var in cat_vars:
    #print(f"\nОбработка переменной: {var}")

    # Проверяем, задана ли базовая категория
    base_cat = base_categories.get(var)

    # Выводим информацию о переменной
    unique_vals = df[var].astype(str).unique()
    #print(f"  Уникальных значений: {len(unique_vals)}")
    #print(f"  Значения: {sorted(unique_vals)}")

    if base_cat:
        #print(f"  Заданная базовая категория: '{base_cat}'")
        # Преобразуем базовую категорию в строку
        base_cat = str(base_cat)

        if base_cat in unique_vals:
            df_encoded = create_dummies_with_base(df_encoded, var, base_cat)
        else:
            #print(f"  ⚠️  ОШИБКА: Базовая категория '{base_cat}' не найдена в данных!")
            #print(f"  Использую первую категорию по алфавиту как базовую")
            df_encoded = create_dummies_with_base(df_encoded, var, None)
    else:
        #print(f"  Базовая категория не задана, использую первую по алфавиту")
        df_encoded = create_dummies_with_base(df_encoded, var, None)

print("\n" + "=" * 80)
print("АНАЛИЗ КОРРЕЛЯЦИЙ МЕЖДУ ПЕРЕМЕННЫМИ")
print("=" * 80)

# Создаем матрицу корреляций
corr_matrix = df_encoded.corr()

# Находим пары с высокой корреляцией
print("\nСамые высокие корреляции (абсолютное значение > 0.7):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_val
            ))

if high_corr_pairs:
    for var1, var2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {var1:30} ↔ {var2:30}: {corr:.3f}")
else:
    print("  Нет пар с корреляцией > 0.7")

print("\nСамые высокие корреляции (абсолютное значение > 0.5):")
moderate_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            moderate_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_val
            ))

if moderate_corr_pairs:
    print(f"  Найдено {len(moderate_corr_pairs)} пар с корреляцией > 0.5")
    print("  Топ-10 по абсолютному значению:")
    for var1, var2, corr in sorted(moderate_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
        print(f"    {var1:30} ↔ {var2:30}: {corr:.3f}")
else:
    print("  Нет пар с корреляцией > 0.5")

# Визуализация корреляционной матрицы (только для наиболее важных переменных)
print("\nСоздание визуализации корреляционной матрицы...")

# Выбираем топ переменных для визуализации (по количеству)
if len(df_encoded.columns) > 30:
    # Если много переменных, выбираем наиболее информативные
    var_importance = {}
    for col in df_encoded.columns:
        # Простая эвристика: переменные с большей дисперсией или небинарные
        if col in quant_vars:
            var_importance[col] = df_encoded[col].var()
        elif col in binary_vars:
            var_importance[col] = 0.5  # Среднее значение для бинарных
        else:
            # Для дамми-переменных используем частоту
            var_importance[col] = df_encoded[col].mean()

    # Сортируем по важности
    top_vars = sorted(var_importance, key=var_importance.get, reverse=True)[:30]
    corr_matrix_vis = df_encoded[top_vars].corr()
else:
    corr_matrix_vis = corr_matrix

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix_vis, dtype=bool))
sns.heatmap(corr_matrix_vis, mask=mask, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot=False, fmt=".2f")
plt.title('Корреляционная матрица переменных', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ===========================================
# 6. РАСЧЕТ VIF ДЛЯ ВСЕХ ПЕРЕМЕННЫХ
# ===========================================

print("\n" + "=" * 80)
print("РАСЧЕТ VIF (КОЭФФИЦИЕНТ ИНФЛЯЦИИ ДИСПЕРСИИ)")
print("=" * 80)

# Для расчета VIF нужна целевая переменная
# Если есть целевая переменная, используем ее, иначе создаем фиктивную
if df_target is not None:
    df_for_vif = df_encoded.copy()
    df_for_vif['severity'] = df_target

    # Удаляем строки с пропусками в целевой переменной
    df_for_vif = df_for_vif.dropna(subset=['severity'])
    print(f"Размер данных для VIF анализа: {df_for_vif.shape}")

    # Выбираем только предикторы
    X = df_for_vif.drop('severity', axis=1)
else:
    print("Целевая переменная не найдена, создаю фиктивную для анализа VIF")
    X = df_encoded.copy()



# Добавляем константу
X_with_const = sm.add_constant(X)

# Расчет VIF
print("\nРасчет VIF для всех переменных...")
vif_data = pd.DataFrame()
vif_data["Variable"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                   for i in range(X_with_const.shape[1])]

# Убираем константу и сортируем
vif_data = vif_data[vif_data['Variable'] != 'const']
vif_data = vif_data.sort_values('VIF', ascending=False)
vif_data['sqrt(VIF)'] = np.sqrt(vif_data['VIF'])
vif_data['Tolerance'] = 1 / vif_data['VIF']

print("\nVIF для всех переменных (отсортировано по убыванию VIF):")
print(vif_data.to_string())



print("\n" + "=" * 80)
print("ГРУППИРОВАННЫЙ АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
print("=" * 80)

# Определяем группы дамми-переменных
dummy_groups = {}
for var in cat_vars:
    dummy_cols = [col for col in df_encoded.columns if col.startswith(f'{var}_')]
    if dummy_cols:
        dummy_groups[var] = dummy_cols


# Анализ VIF по группам
print("\nСредний VIF по группам переменных:")
group_stats = []
for group, vars_list in dummy_groups.items():
    if vars_list:
        group_vif = vif_data[vif_data['Variable'].isin(vars_list)]
        if len(group_vif) > 0:
            mean_vif = group_vif['VIF'].mean()
            max_vif = group_vif['VIF'].max()
            group_stats.append({
                'Группа': group,
                'Кол-во': len(vars_list),
                'Ср. VIF': mean_vif,
                'Макс. VIF': max_vif,
            })
 