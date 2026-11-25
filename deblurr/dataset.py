import glob, random, cv2, numpy as np, torch
from torch.utils.data import Dataset

def make_blur(img, rng):
    k = int(rng.choice([5, 9, 13, 17]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.randint(0, 180), 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)

class DeblurDataset(Dataset):
    def __init__(self, folder, crop=128, train=True):
        self.files = sorted(glob.glob(f"{folder}/sharp/*.jpg"))
        assert len(self.files) > 0, f"No images in {folder}/sharp"
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        assert img is not None, f"Could not read {self.files[i]}"
        c = self.crop
        h, w = img.shape[:2]
        if h < c or w < c:
            img = cv2.resize(img, (max(w, c), max(h, c)))
            h, w = img.shape[:2]

        # train: random blur every time; val: fixed blur per image index
        rng = np.random if self.train else np.random.RandomState(i)
        blurred = make_blur(img, rng)

        if self.train:
            y, x = random.randint(0, h - c), random.randint(0, w - c)
        else:
            y, x = (h - c) // 2, (w - c) // 2
        sharp_c = img[y:y + c, x:x + c]
        blur_c = blurred[y:y + c, x:x + c]

        if self.train and random.random() < 0.5:
            sharp_c, blur_c = sharp_c[:, ::-1], blur_c[:, ::-1]

        def to_tensor(a):
            a = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_BGR2RGB)
            return torch.from_numpy(a).permute(2, 0, 1).float() / 255.0

        return to_tensor(blur_c), to_tensor(sharp_c)