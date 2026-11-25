import os, glob, random, shutil, cv2, numpy as np

SRC = "dataset/val2017/val2017"      # change if your path differs
OUT = "data"

random.seed(42)
np.random.seed(42)

files = sorted(glob.glob(f"{SRC}/*.jpg"))
print("Images found:", len(files))
assert len(files) == 5000, "Expected 5000 images, check SRC path"

random.shuffle(files)
splits = {"train": files[:4000], "val": files[4000:4500], "test": files[4500:]}

for name, fl in splits.items():
    os.makedirs(f"{OUT}/{name}/sharp", exist_ok=True)
    for f in fl:
        shutil.copy(f, f"{OUT}/{name}/sharp/")

def make_blur(img):
    k = int(np.random.choice([5, 9, 13, 17]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), np.random.randint(0, 180), 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

# Fixed blurred test set (same result every run because of the seed)
os.makedirs(f"{OUT}/test/blurred", exist_ok=True)
for f in sorted(glob.glob(f"{OUT}/test/sharp/*.jpg")):
    img = cv2.imread(f)
    cv2.imwrite(f"{OUT}/test/blurred/{os.path.basename(f)}", make_blur(img))

print({k: len(v) for k, v in splits.items()})