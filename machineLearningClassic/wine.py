## Use Select KBest to select N amount of features.
## test this features using the same classifier you 

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
X,y = load_digits(return_X_y=True)
X_new = SelectKBest(chi2, k=20).fit_transform(X,y)


