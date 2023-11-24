from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class DataGenerator:
    n_rows: int
    categories: list
    p: float
    loc1: float
    loc2: float
    scale1: float
    scale2: float

    def generate_data(self):
        half_n_rows = self.n_rows // 2
        bool_col_systemic = np.random.choice([True, False], size=half_n_rows, p=[self.p, 1 - self.p])
        bool_col_no_systemic = np.random.choice([True, False], size=half_n_rows, p=[1 - self.p, self.p])

        float_col_systemic = np.random.normal(loc=self.loc1, scale=self.scale1, size=half_n_rows).astype(np.float32)
        float_col_no_systemic = np.random.normal(loc=self.loc2, scale=self.scale2, size=half_n_rows).astype(np.float32)

        dir1 = np.random.dirichlet(np.ones(len(self.categories)), size=1)[0]
        dir2 = np.random.dirichlet(np.ones(len(self.categories)), size=1)[0]
        cat_col_systemic = np.random.choice(self.categories, size=half_n_rows, p=dir1)
        cat_col_no_systemic = np.random.choice(self.categories, size=half_n_rows, p=dir2)

        data = pd.DataFrame({
            "float_feature": np.concatenate([float_col_systemic, float_col_no_systemic]),
            "bool_feature": np.concatenate([bool_col_systemic, bool_col_no_systemic]),
            "category_feature": pd.Categorical(np.concatenate([cat_col_systemic, cat_col_no_systemic]), categories=self.categories, ordered=True),
            "is_systemic_risk": np.array([True] * half_n_rows + [False] * half_n_rows)
        })

        data['verbatim_text'] = data['is_systemic_risk'].apply(self.generate_verbatim)

        # Add a date column
        date_range = pd.date_range(start='2022-01-01', end='2022-12-31', periods=self.n_rows)
        data['date'] = np.random.choice(date_range, size=self.n_rows)

        return data

    def generate_verbatim(self, is_systemic):
        p = self.p
        if is_systemic:
            p = 1 - p
        return_systemic = np.random.choice([True, False], p=[p, 1 - p])

        if return_systemic:
            topics = ["security breach", "rate changes", "hidden fees", "misleading schemes", "service responsiveness"]
        else:
            topics = ["delayed response", "minor errors", "application processing", "terms clarification", "platform usability"]
        issue = np.random.choice(topics)
        return f"Customer complained about {issue}."
