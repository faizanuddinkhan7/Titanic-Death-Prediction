"""
Data Preprocessing and Feature Engineering Pipeline for Titanic Dataset.
Uses pandas and numpy for robust data cleaning, feature engineering,
and scaling without data leakage.
"""

import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler

class TitanicPreprocessor:
    def __init__(self):
        self.age_medians_ = {}
        self.overall_age_median_ = 28.0
        self.fare_medians_ = {}
        self.overall_fare_median_ = 14.45
        self.embarked_mode_ = 'S'
        self.ticket_counts_ = {}
        self.scaler = StandardScaler()
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    @staticmethod
    def _extract_title(name: str) -> str:
        """Extract title from passenger name and group rare titles."""
        match = re.search(r' ([A-Za-z]+)\.', name)
        if not match:
            return 'Rare'
        title = match.group(1)
        if title in ['Mr', 'Miss', 'Mrs', 'Master']:
            return title
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title in ['Mme']:
            return 'Mrs'
        else:
            return 'Rare'

    @staticmethod
    def _extract_deck(cabin: Optional[str]) -> str:
        """Extract deck letter from cabin code or assign 'M' for missing."""
        if pd.isna(cabin) or not str(cabin).strip():
            return 'M'
        deck = str(cabin).strip()[0].upper()
        if deck not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            return 'M'
        return deck

    def fit(self, df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> 'TitanicPreprocessor':
        """
        Fit preprocessing statistics on training data (and ticket counts combined).
        """
        data = df.copy()
        
        # 1. Mode for Embarked
        self.embarked_mode_ = data['Embarked'].mode()[0] if not data['Embarked'].dropna().empty else 'S'

        # 2. Titles for median age calculation
        data['Title'] = data['Name'].apply(self._extract_title)
        
        # Overall medians
        self.overall_age_median_ = float(data['Age'].median()) if not data['Age'].dropna().empty else 28.0
        self.overall_fare_median_ = float(data['Fare'].median()) if not data['Fare'].dropna().empty else 14.45

        # Group median age by (Pclass, Title)
        age_grouped = data.groupby(['Pclass', 'Title'])['Age'].median()
        self.age_medians_ = age_grouped.to_dict()

        # Group median fare by (Pclass, Embarked)
        data_embarked = data['Embarked'].fillna(self.embarked_mode_)
        fare_grouped = data.groupby([data['Pclass'], data_embarked])['Fare'].median()
        self.fare_medians_ = fare_grouped.to_dict()

        # Ticket counts across all known tickets (train + test if available)
        combined_tickets = data['Ticket'].astype(str)
        if test_df is not None:
            combined_tickets = pd.concat([combined_tickets, test_df['Ticket'].astype(str)], ignore_index=True)
        self.ticket_counts_ = combined_tickets.value_counts().to_dict()

        # Fit feature matrix to determine feature columns and scaler
        X, _ = self._engineer_features(data, is_fit=True)
        
        # Continuous columns to scale
        cont_cols = ['Age', 'LogFare', 'FamilySize', 'TicketFreq']
        self.scaler.fit(X[cont_cols])
        
        self.feature_names_ = self._get_final_feature_names(X, cont_cols)
        self.is_fitted = True
        return self

    def _engineer_features(self, df: pd.DataFrame, is_fit: bool = False) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Extract engineered features and handle missing values."""
        df_out = pd.DataFrame(index=df.index)

        # 1. Title
        title_series = df['Name'].apply(self._extract_title)
        df_out['Title'] = title_series

        # 2. Embarked
        df_out['Embarked'] = df['Embarked'].fillna(self.embarked_mode_)

        # 3. Pclass
        df_out['Pclass'] = df['Pclass']

        # 4. Sex (binary: female=1, male=0)
        df_out['IsFemale'] = (df['Sex'] == 'female').astype(float)

        # 5. Missing Age imputation using (Pclass, Title) median
        def impute_age(row):
            if pd.notna(row['Age']):
                return float(row['Age'])
            key = (row['Pclass'], row['Title'])
            return self.age_medians_.get(key, self.overall_age_median_)

        temp_df = df[['Pclass', 'Age']].copy()
        temp_df['Title'] = title_series
        df_out['Age'] = temp_df.apply(impute_age, axis=1)

        # 6. Missing Fare imputation using (Pclass, Embarked) median
        def impute_fare(row):
            if pd.notna(row['Fare']):
                return float(row['Fare'])
            key = (row['Pclass'], row['Embarked'])
            return self.fare_medians_.get(key, self.overall_fare_median_)

        temp_df_fare = df[['Pclass', 'Fare']].copy()
        temp_df_fare['Embarked'] = df_out['Embarked']
        fare_clean = temp_df_fare.apply(impute_fare, axis=1)
        df_out['LogFare'] = np.log1p(np.maximum(0.0, fare_clean))

        # 7. Family Size and IsAlone
        sibsp = df['SibSp'].fillna(0).astype(int)
        parch = df['Parch'].fillna(0).astype(int)
        family_size = sibsp + parch + 1
        df_out['FamilySize'] = family_size.astype(float)
        df_out['IsAlone'] = (family_size == 1).astype(float)

        # Family group categories: Alone (1), Small (2-4), Large (5+)
        def fam_group(size):
            if size == 1:
                return 'Alone'
            elif 2 <= size <= 4:
                return 'Small'
            else:
                return 'Large'
        df_out['FamilyGroup'] = family_size.apply(fam_group)

        # 8. Cabin Deck
        df_out['CabinDeck'] = df['Cabin'].apply(self._extract_deck)

        # 9. Ticket Frequency
        df_out['TicketFreq'] = df['Ticket'].astype(str).map(lambda t: self.ticket_counts_.get(t, 1)).astype(float)

        # 10. Interaction feature: Child (Age < 12)
        df_out['IsChild'] = (df_out['Age'] < 12.0).astype(float)
        
        # 11. Interaction feature: Mother (female, Parch > 0, Age > 18, Title != 'Miss')
        is_mother = (df['Sex'] == 'female') & (parch > 0) & (df_out['Age'] > 18.0) & (title_series == 'Mrs')
        df_out['IsMother'] = is_mother.astype(float)

        y = df['Survived'].values.astype(np.float32) if 'Survived' in df.columns else None
        return df_out, y

    def _get_final_feature_names(self, X_eng: pd.DataFrame, cont_cols: List[str]) -> List[str]:
        """Determine fixed feature column list after one-hot encoding."""
        cat_cols = ['Pclass', 'Title', 'Embarked', 'CabinDeck', 'FamilyGroup']
        binary_cols = ['IsFemale', 'IsAlone', 'IsChild', 'IsMother']
        
        feature_names = list(cont_cols) + list(binary_cols)
        
        # Add categorical dummy column names
        dummy_df = pd.get_dummies(X_eng[cat_cols], columns=cat_cols, drop_first=False)
        for col in sorted(dummy_df.columns):
            feature_names.append(col)
        return feature_names

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform raw DataFrame into scaled numeric numpy feature matrix and targets.
        """
        assert self.is_fitted, "TitanicPreprocessor must be fitted before calling transform."
        X_eng, y = self._engineer_features(df, is_fit=False)
        
        cont_cols = ['Age', 'LogFare', 'FamilySize', 'TicketFreq']
        binary_cols = ['IsFemale', 'IsAlone', 'IsChild', 'IsMother']
        cat_cols = ['Pclass', 'Title', 'Embarked', 'CabinDeck', 'FamilyGroup']

        # Scale continuous features
        X_cont_scaled = self.scaler.transform(X_eng[cont_cols])
        X_cont_df = pd.DataFrame(X_cont_scaled, columns=cont_cols, index=df.index)

        # Binary features
        X_bin_df = X_eng[binary_cols]

        # Categorical dummies
        X_cat_df = pd.get_dummies(X_eng[cat_cols], columns=cat_cols, drop_first=False)

        # Combine
        combined_df = pd.concat([X_cont_df, X_bin_df, X_cat_df], axis=1)

        # Align with fitted feature_names_ (add missing cols with 0.0, drop unexpected cols)
        for col in self.feature_names_:
            if col not in combined_df.columns:
                combined_df[col] = 0.0
        
        aligned_df = combined_df[self.feature_names_].astype(np.float32)
        X_array = aligned_df.values
        return X_array, y

    def get_feature_names(self) -> List[str]:
        return self.feature_names_

