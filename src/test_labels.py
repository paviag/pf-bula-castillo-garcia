
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg

PATH = "datasets/dataset"
print(os.getcwd())

for split in ["train", "val"]:
    print(f"Checking {split} folder")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Images in {split} folder")
    axes = axes.flatten()

    image_folder = os.path.join(PATH, "images", split)
    label_folder = os.path.join(PATH, "labels", split)
    image_files = os.listdir(image_folder)

    for i, image_file in enumerate(image_files[:9]):  # Display up to 9 images
        img_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.split('.')[0] + ".txt")

        img = mpimg.imread(img_path)
        h, w = img.shape[:2]

        axes[i].imshow(img)
        axes[i].set_title(image_file.split(".")[0])
        axes[i].axis("off")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: unexpected format in {label_path}: {line}")
                continue

            _, bx, by, bw, bh = map(float, parts)
            if not (0 <= bx <= 1 and 0 <= by <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                print(f"Normalization error in {image_file}: {line}")

            # Convert YOLO format to xmin, ymin
            xmin = (bx - bw / 2) * w
            ymin = (by - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            rect = Rectangle((xmin, ymin), box_w, box_h,
                             edgecolor="red", facecolor="none", linewidth=2)
            axes[i].add_patch(rect)

    fig.tight_layout()
    plt.show()
