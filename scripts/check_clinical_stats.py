import sys
sys.path.insert(0, '.')

from src.data.clinical_features import build_clinical_features
import pandas as pd

train_df = pd.read_parquet('data/processed/image_subset_manifest.parquet')
train_df = train_df[train_df['split'] == 'train']
features_df, cols = build_clinical_features(train_df, train_frame=train_df)

print('=== AGE ===')
print(f'  min: {features_df["age"].min():.4f}')
print(f'  max: {features_df["age"].max():.4f}')
print(f'  mean: {features_df["age"].mean():.4f}')
print(f'  std: {features_df["age"].std():.4f}')
print()

print('=== RECENT_BMI ===')
print(f'  min: {features_df["recent_bmi"].min():.4f}')
print(f'  max: {features_df["recent_bmi"].max():.4f}')
print(f'  mean: {features_df["recent_bmi"].mean():.4f}')
print(f'  std: {features_df["recent_bmi"].std():.4f}')
print()

print('=== FLAG COLUMNS (0-1 range) ===')
flag_cols = [c for c in cols if c not in ['age', 'recent_bmi']]
for c in flag_cols[:5]:
    print(f'  {c}: min={features_df[c].min()}, max={features_df[c].max()}, mean={features_df[c].mean():.4f}')
print('  ...')
print(f'Total feature columns: {len(cols)}')