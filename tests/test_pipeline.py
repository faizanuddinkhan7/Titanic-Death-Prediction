"""
Aggressive Test Suite for Titanic ML Pipeline.
Tests data cleaning, preprocessing, PyTorch model mechanics,
trained fold weights, and submission integrity.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import TitanicPreprocessor
from src.dataset import TitanicDataset, create_dataloaders
from src.model import TitanicMLP

class TestTitanicDataCleaningAndPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df = pd.read_csv("titanic/train.csv")
        cls.test_df = pd.read_csv("titanic/test.csv")
        cls.preprocessor = TitanicPreprocessor()
        cls.preprocessor.fit(cls.train_df, cls.test_df)
        cls.X_train, cls.y_train = cls.preprocessor.transform(cls.train_df)
        cls.X_test, cls.y_test = cls.preprocessor.transform(cls.test_df)

    def test_no_missing_values_after_preprocessing(self):
        """Ensure no NaNs exist in train or test matrices."""
        self.assertEqual(np.isnan(self.X_train).sum(), 0, "Train feature matrix contains NaNs")
        self.assertEqual(np.isnan(self.y_train).sum(), 0, "Train target array contains NaNs")
        self.assertEqual(np.isnan(self.X_test).sum(), 0, "Test feature matrix contains NaNs")

    def test_dimensions_match(self):
        """Feature dimensionality must match exactly between train and test."""
        self.assertEqual(self.X_train.shape[1], self.X_test.shape[1])
        self.assertEqual(self.X_train.shape[0], 891)
        self.assertEqual(self.X_test.shape[0], 418)
        self.assertEqual(len(self.preprocessor.get_feature_names()), self.X_train.shape[1])

    def test_title_extraction(self):
        """Test title extraction and regrouping logic."""
        self.assertEqual(TitanicPreprocessor._extract_title("Braund, Mr. Owen Harris"), "Mr")
        self.assertEqual(TitanicPreprocessor._extract_title("Cumings, Mrs. John Bradley"), "Mrs")
        self.assertEqual(TitanicPreprocessor._extract_title("Heikkinen, Miss. Laina"), "Miss")
        self.assertEqual(TitanicPreprocessor._extract_title("Palsson, Master. Gosta"), "Master")
        self.assertEqual(TitanicPreprocessor._extract_title("Reynaldo, Ms. Encarnacion"), "Miss")
        self.assertEqual(TitanicPreprocessor._extract_title("Aubart, Mme. Leontine"), "Mrs")
        self.assertEqual(TitanicPreprocessor._extract_title("Dr. Minahan, William Edward"), "Rare")

    def test_cabin_deck_extraction(self):
        """Test cabin deck extraction."""
        self.assertEqual(TitanicPreprocessor._extract_deck("C85"), "C")
        self.assertEqual(TitanicPreprocessor._extract_deck("B96 B98"), "B")
        self.assertEqual(TitanicPreprocessor._extract_deck("F G63"), "F")
        self.assertEqual(TitanicPreprocessor._extract_deck(None), "M")
        self.assertEqual(TitanicPreprocessor._extract_deck(""), "M")

    def test_continuous_scaling_properties(self):
        """Check scaled continuous features have approximately 0 mean and unit variance."""
        cont_features = self.X_train[:, :4]  # Age, LogFare, FamilySize, TicketFreq
        means = np.mean(cont_features, axis=0)
        stds = np.std(cont_features, axis=0)
        for i, (m, s) in enumerate(zip(means, stds)):
            self.assertAlmostEqual(m, 0.0, delta=0.1, msg=f"Feature {i} mean not near 0")
            self.assertAlmostEqual(s, 1.0, delta=0.1, msg=f"Feature {i} std not near 1")

class TestTitanicModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.input_dim = 30
        self.model = TitanicMLP(input_dim=self.input_dim, hidden_dims=[64, 32], dropout=0.2)

    def test_forward_output_shape(self):
        """Forward pass should produce [batch_size, 1] tensor in both train and eval modes."""
        self.model.eval()
        for batch_size in [1, 16, 64]:
            x = torch.randn(batch_size, self.input_dim)
            out = self.model(x)
            self.assertEqual(out.shape, (batch_size, 1))

        self.model.train()
        for batch_size in [2, 16, 64]:
            x = torch.randn(batch_size, self.input_dim)
            out = self.model(x)
            self.assertEqual(out.shape, (batch_size, 1))

    def test_gradient_flow(self):
        """Check all trainable parameters receive non-zero gradients during backprop."""
        x = torch.randn(16, self.input_dim)
        y = torch.tensor([[1.0]] * 8 + [[0.0]] * 8)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        logits = self.model(x)
        loss = criterion(logits, y)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
                self.assertTrue(torch.any(param.grad != 0), f"Parameter {name} has strictly zero gradient")

    def test_probabilities_and_predictions(self):
        """Check predict_proba in [0, 1] and predict produces {0, 1}."""
        x = torch.randn(20, self.input_dim)
        probs = self.model.predict_proba(x)
        preds = self.model.predict(x)

        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities outside [0, 1]")
        unique_preds = torch.unique(preds).tolist()
        for p in unique_preds:
            self.assertIn(p, [0, 1], f"Prediction {p} is not binary")

class TestCrossValidationAndTrainedArtifacts(unittest.TestCase):
    def test_fold_models_exist_and_loadable(self):
        """Verify all 5 fold models exist, can be loaded, and perform inference."""
        models_dir = "models"
        self.assertTrue(os.path.exists(models_dir), "Models directory does not exist")

        test_tensor = torch.randn(10, 30)
        for fold in range(1, 6):
            model_file = os.path.join(models_dir, f"titanic_mlp_fold{fold}.pth")
            self.assertTrue(os.path.exists(model_file), f"Fold model {model_file} not found")

            model = TitanicMLP(input_dim=30, hidden_dims=[64, 32], dropout=0.25)
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            with torch.no_grad():
                preds = model(test_tensor)
                self.assertEqual(preds.shape, (10, 1))

    def test_plots_exist(self):
        """Verify EDA and training curve visualization plots exist."""
        self.assertTrue(os.path.exists("plots/eda_summary.png"), "plots/eda_summary.png is missing")
        self.assertTrue(os.path.exists("plots/training_curves.png"), "plots/training_curves.png is missing")

class TestSubmissionIntegrity(unittest.TestCase):
    def setUp(self):
        self.sub_path = "submission.csv"
        self.test_df = pd.read_csv("titanic/test.csv")

    def test_submission_file_validity(self):
        """Verify competition submission specifications."""
        self.assertTrue(os.path.exists(self.sub_path), "submission.csv does not exist")
        df_sub = pd.read_csv(self.sub_path)

        # 1. Exactly 418 rows
        self.assertEqual(len(df_sub), 418, f"Expected 418 rows but got {len(df_sub)}")

        # 2. Exactly 2 columns: PassengerId and Survived
        self.assertListEqual(list(df_sub.columns), ['PassengerId', 'Survived'])

        # 3. No missing or null values
        self.assertEqual(df_sub.isnull().sum().sum(), 0, "Submission contains null values")

        # 4. Correct PassengerId alignment
        self.assertTrue((df_sub['PassengerId'].values == self.test_df['PassengerId'].values).all(),
                        "PassengerId column does not match test.csv exactly")

        # 5. Survived contains only 0 or 1
        valid_values = set(df_sub['Survived'].unique())
        self.assertTrue(valid_values.issubset({0, 1}), f"Survived column contains invalid values: {valid_values}")

        # 6. Sanity check on survival rate: between 25% and 50%
        survival_rate = df_sub['Survived'].mean()
        self.assertGreater(survival_rate, 0.25, f"Survival rate {survival_rate:.2f} is unrealistically low")
        self.assertLess(survival_rate, 0.50, f"Survival rate {survival_rate:.2f} is unrealistically high")

if __name__ == "__main__":
    unittest.main(verbosity=2)
