"""
Exploratory Data Analysis (EDA) for Titanic Dataset.
Uses pandas, numpy, and matplotlib to analyze the distributions,
missing data patterns, and survival correlations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_eda(train_path: str = "titanic/train.csv", 
            test_path: str = "titanic/test.csv",
            output_dir: str = "plots") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("=" * 60)
    print("TITANIC DATASET OVERVIEW")
    print("=" * 60)
    print(f"Training set: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Testing set:  {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    # Missing values
    train_missing = train_df.isnull().sum()
    train_missing_pct = (train_missing / len(train_df)) * 100
    test_missing = test_df.isnull().sum()
    test_missing_pct = (test_missing / len(test_df)) * 100

    print("\nTraining Missing Values:")
    for col in train_missing[train_missing > 0].index:
        print(f"  - {col}: {train_missing[col]} ({train_missing_pct[col]:.2f}%)")

    print("\nTesting Missing Values:")
    for col in test_missing[test_missing > 0].index:
        print(f"  - {col}: {test_missing[col]} ({test_missing_pct[col]:.2f}%)")

    # Overall Survival Rate
    survived_counts = train_df['Survived'].value_counts()
    overall_survival_rate = train_df['Survived'].mean() * 100
    print(f"\nOverall Survival Rate: {overall_survival_rate:.2f}% ({survived_counts[1]} survived, {survived_counts[0]} died)")

    # Sex Survival Rate
    sex_survival = train_df.groupby('Sex')['Survived'].agg(['count', 'mean'])
    print("\nSurvival by Sex:")
    for sex, row in sex_survival.iterrows():
        print(f"  - {sex.capitalize()}: {row['mean']*100:.2f}% survival (total {int(row['count'])})")

    # Pclass Survival Rate
    pclass_survival = train_df.groupby('Pclass')['Survived'].agg(['count', 'mean'])
    print("\nSurvival by Passenger Class (Pclass):")
    for pclass, row in pclass_survival.iterrows():
        print(f"  - Class {pclass}: {row['mean']*100:.2f}% survival (total {int(row['count'])})")

    # Extract Title for quick analysis
    train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_survival = train_df.groupby('Title')['Survived'].agg(['count', 'mean']).sort_values(by='count', ascending=False)
    print("\nTop Titles Survival:")
    for title, row in title_survival.head(6).iterrows():
        print(f"  - {title}: {row['mean']*100:.2f}% survival (total {int(row['count'])})")

    # Family Size
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    family_survival = train_df.groupby('FamilySize')['Survived'].agg(['count', 'mean'])
    print("\nSurvival by Family Size:")
    for fam_sz, row in family_survival.iterrows():
        print(f"  - Size {fam_sz}: {row['mean']*100:.2f}% survival (total {int(row['count'])})")

    # Visualizations using matplotlib
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Titanic Dataset Exploratory Data Analysis', fontsize=18, fontweight='bold')

    # 1. Overall Survival Count
    axes[0, 0].bar(['Deceased (0)', 'Survived (1)'], 
                   [survived_counts[0], survived_counts[1]], 
                   color=['#d9534f', '#5cb85c'], edgecolor='black', width=0.55)
    axes[0, 0].set_title(f'Overall Survival (Rate: {overall_survival_rate:.1f}%)', fontsize=12, fontweight='semibold')
    axes[0, 0].set_ylabel('Number of Passengers')
    for i, v in enumerate([survived_counts[0], survived_counts[1]]):
        axes[0, 0].text(i, v + 10, f"{v} ({v/len(train_df)*100:.1f}%)", ha='center', fontweight='bold')

    # 2. Survival by Sex
    sexes = ['Female', 'Male']
    sex_surv_rates = [train_df[train_df['Sex'] == 'female']['Survived'].mean() * 100,
                      train_df[train_df['Sex'] == 'male']['Survived'].mean() * 100]
    axes[0, 1].bar(sexes, sex_surv_rates, color=['#e83e8c', '#007bff'], edgecolor='black', width=0.55)
    axes[0, 1].set_title('Survival Rate by Sex', fontsize=12, fontweight='semibold')
    axes[0, 1].set_ylabel('Survival Rate (%)')
    axes[0, 1].set_ylim(0, 100)
    for i, v in enumerate(sex_surv_rates):
        axes[0, 1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    # 3. Survival by Class (Pclass)
    pclasses = ['1st Class', '2nd Class', '3rd Class']
    pclass_rates = [train_df[train_df['Pclass'] == c]['Survived'].mean() * 100 for c in [1, 2, 3]]
    axes[0, 2].bar(pclasses, pclass_rates, color=['#ffc107', '#17a2b8', '#6c757d'], edgecolor='black', width=0.55)
    axes[0, 2].set_title('Survival Rate by Ticket Class', fontsize=12, fontweight='semibold')
    axes[0, 2].set_ylabel('Survival Rate (%)')
    axes[0, 2].set_ylim(0, 100)
    for i, v in enumerate(pclass_rates):
        axes[0, 2].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    # 4. Age Distribution by Survival
    survived_ages = train_df[train_df['Survived'] == 1]['Age'].dropna()
    deceased_ages = train_df[train_df['Survived'] == 0]['Age'].dropna()
    bins = np.linspace(0, 80, 25)
    axes[1, 0].hist(deceased_ages, bins=bins, alpha=0.6, label='Deceased', color='#d9534f', edgecolor='black')
    axes[1, 0].hist(survived_ages, bins=bins, alpha=0.6, label='Survived', color='#5cb85c', edgecolor='black')
    axes[1, 0].set_title('Age Distribution by Survival Status', fontsize=12, fontweight='semibold')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Passenger Count')
    axes[1, 0].legend()

    # 5. Survival Rate by Family Size
    fam_sizes = sorted(train_df['FamilySize'].unique())
    fam_rates = [train_df[train_df['FamilySize'] == sz]['Survived'].mean() * 100 for sz in fam_sizes]
    fam_counts = [len(train_df[train_df['FamilySize'] == sz]) for sz in fam_sizes]
    axes[1, 1].plot(fam_sizes, fam_rates, marker='o', linewidth=2, color='#6f42c1', markersize=8)
    axes[1, 1].set_title('Survival Rate vs Family Size (SibSp + Parch + 1)', fontsize=12, fontweight='semibold')
    axes[1, 1].set_xlabel('Family Size')
    axes[1, 1].set_ylabel('Survival Rate (%)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    for sz, rate, cnt in zip(fam_sizes, fam_rates, fam_counts):
        axes[1, 1].text(sz, rate + 3, f"{rate:.0f}% (n={cnt})", ha='center', fontsize=8)

    # 6. Fare Distribution by Survival (log-scale)
    survived_fares = np.log1p(train_df[train_df['Survived'] == 1]['Fare'])
    deceased_fares = np.log1p(train_df[train_df['Survived'] == 0]['Fare'])
    axes[1, 2].boxplot([deceased_fares, survived_fares], tick_labels=['Deceased', 'Survived'], patch_artist=True,
                       boxprops=dict(facecolor='#17a2b8', alpha=0.6))
    axes[1, 2].set_title('Log(Fare + 1) Distribution by Survival', fontsize=12, fontweight='semibold')
    axes[1, 2].set_ylabel('Log(Fare + 1)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    eda_plot_path = os.path.join(output_dir, 'eda_summary.png')
    plt.savefig(eda_plot_path, dpi=150)
    plt.close()
    print(f"\nEDA summary plot saved to: {eda_plot_path}")

    return {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "overall_survival_rate": overall_survival_rate,
        "plot_path": eda_plot_path
    }

if __name__ == "__main__":
    run_eda()

