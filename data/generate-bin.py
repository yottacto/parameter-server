import numpy as np

data = np.loadtxt("pubmed.feature")
np.save("pubmed.feature", data)

data = np.loadtxt("pubmed.label")
np.save("pubmed.label", data)

