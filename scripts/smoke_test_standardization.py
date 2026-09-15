import sys
sys.path.insert(0, '.')

from src.data.clinical_dataset import ClinicalDataset
from src.data.clinical_features import build_clinical_features, get_train_standardization_stats
import pandas as pd

print('=== TRAIN ===')
train_df = pd.read_parquet('data/processed/image_subset_manifest.parquet')
train_df = train_df[train_df['split'] == 'train']

train_age_mean, train_age_std, train_bmi_mean, train_bmi_std = get_train_standardization_stats(train_df)
print('Train age_mean={:.4f}, age_std={:.4f}'.format(train_age_mean, train_age_std))
print('Train bmi_mean={:.4f}, bmi_std={:.4f}'.format(train_bmi_mean, train_bmi_std))

features_df, _ = build_clinical_features(train_df, train_frame=train_df)
print('Train age (standardized): mean={:.6f}, std={:.6f}'.format(features_df['age'].mean(), features_df['age'].std()))
print('Train bmi (standardized): mean={:.6f}, std={:.6f}'.format(features_df['recent_bmi'].mean(), features_df['recent_bmi'].std()))
print()

print('=== VAL ===')
val_df = pd.read_parquet('data/processed/image_subset_manifest.parquet')
val_df = val_df[val_df['split'] == 'val']
train_df = pd.read_parquet('data/processed/image_subset_manifest.parquet')
train_df = train_df[train_df['split'] == 'train']

features_df, _ = build_clinical_features(
    val_df,
    train_frame=train_df,
    age_mean=train_age_mean, age_std=train_age_std,
    bmi_mean=train_bmi_mean, bmi_std=train_bmi_std
)
print('Val age (standardized with train stats): mean={:.6f}, std={:.6f}'.format(features_df['age'].mean(), features_df['age'].std()))
print('Val bmi (standardized with train stats): mean={:.6f}, std={:.6f}'.format(features_df['recent_bmi'].mean(), features_df['recent_bmi'].std()))
print()

print('=== DATASET CLASSES ===')
train_ds = ClinicalDataset('data/processed/image_subset_manifest.parquet', split='train')
val_ds = ClinicalDataset('data/processed/image_subset_manifest.parquet', split='val')

print('Train dataset age mean={:.6f}, std={:.6f}'.format(train_ds.features[:, 0].mean().item(), train_ds.features[:, 0].std().item()))
print('Val dataset age mean={:.6f}, std={:.6f}'.format(val_ds.features[:, 0].mean().item(), val_ds.features[:, 0].std().item()))
print('Train dataset bmi mean={:.6f}, std={:.6f}'.format(train_ds.features[:, 2].mean().item(), train_ds.features[:, 2].std().item()))
print('Val dataset bmi mean={:.6f}, std={:.6f}'.format(val_ds.features[:, 2].mean().item(), val_ds.features[:, 2].std().item()))
print()
print('feature_dim matches: {}'.format(train_ds.feature_dim == val_ds.feature_dim))
print('Smoke test passed')