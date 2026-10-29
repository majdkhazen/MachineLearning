# Machine Learning Coursework

Python coursework exploring classification, model selection, and regression through custom estimators and Jupyter notebooks.

| Folder | Focus | Entry points |
| --- | --- | --- |
| `HW1/` | Data preparation and k-nearest neighbors | `Major_HW1.ipynb`, `prepare.py`, `kNN.py` |
| `HW2/` | Soft-margin SVM and model selection | `Major_HW2.ipynb`, `SoftSVM.py`, `verify_gradients.py` |
| `HW3/` | Linear regression and gradient-based optimization | `Major_HW3.ipynb`, `LinearRegressor.py`, `verify_gradients.py` |

## Explore the notebooks

Use Python 3 with Jupyter and the scientific Python stack:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyter numpy pandas scipy scikit-learn matplotlib
jupyter notebook
```

Open a notebook in its assignment folder so relative dataset paths resolve correctly. Assignment reports and provided datasets are included alongside the code.

## Notes

These are academic implementations with assignment scaffolding and supporting materials. Dependencies are not version-pinned, and the notebooks have not been re-executed as part of this documentation update. The current `HW1/kNN.py` references `np` without importing NumPy; standalone use requires that import.
