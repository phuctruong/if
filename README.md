## 🔍 Step 1: Where to get the data

A reliable public source is **WiggleZ Survey** BAO data on Zenodo:

* **WiggleZ BAO survey files** (galaxy correlation & covariance): available here ([mdpi.com][1], [zenodo.org][2], [arxiv.org][3])

These files include:

* `xi_combined*.dat` – correlation function versus separation (Mpc/h)
* Covariance matrices
* Associated redshift slices

---

## 🧪 Step 2: Code to download in Colab

```python
# Download WiggleZ BAO data
import os
import urllib.request

# Base URL for WiggleZ data on Zenodo
base_url = "https://zenodo.org/record/33470/files/"

files = [
    "xi_combined_z0.2-0.6.dat",
    # Add other redshift ranges if desired
]

os.makedirs("data/bao", exist_ok=True)

for fname in files:
    url = base_url + fname
    out = f"data/bao/{fname}"
    print("Downloading", fname)
    urllib.request.urlretrieve(url, out)
    print("Saved to", out)
```

📁 This downloads the `.dat` files into `data/bao/` for easy parsing.

---

## 🔧 Step 3: Quick parsing example

```python
import numpy as np

# Load one redshift slice
r_sep, xi, xi_err = np.loadtxt("data/bao/xi_combined_z0.2-0.6.dat", unpack=True)

# We'll overlay the separation scale where xi peaks (BAO scale)
bao_scale = r_sep[np.argmax(xi)]
print("Detected BAO at r ≈", bao_scale, "Mpc/h")
```

---

## 📊 Step 4: Overlay in the plot

Modify your existing plot to include:

```python
plt.scatter([bao_scale], [0], marker='x', color='black', label="BAO scale")
```

This marks the observed BAO distance directly on the cosmological **r-axis**, tying real-world measurement to your prime-curvature model.

