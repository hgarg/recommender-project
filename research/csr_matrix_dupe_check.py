import pandas as pd
from scipy.sparse import csr_matrix

# same user + same product, two ratings in the raw data
rows = [0, 0]
cols = [1, 1]
vals = [4, 5]

m = csr_matrix((vals, (rows, cols)), shape=(2, 2))
print(m.toarray())
print("This sums to 9, which is not a valid single rating.")

# check how often duplicates appear in the real dataset
# they need to be collapsed before the matrix is built
raw = pd.read_csv("../data/raw/events.csv")
dupes = raw.duplicated(subset=["user_id", "product_id"]).sum()
print("repeat user-product pairs in the full dataset:", dupes)
