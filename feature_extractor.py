import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_model_and_preprocess(model_name='resnet50'):
    model_name = model_name.lower()
    if model_name == 'resnet50':
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        return base, preprocess_input, (224, 224)

    # fallback
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return base, preprocess_input, (224, 224)

def extract_one(base_model, preprocess_fn, input_size, img_path):
    from tensorflow.keras.preprocessing import image
    try:
        img = image.load_img(img_path, target_size=input_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_fn(x)
        feats = base_model.predict(x, verbose=0)
        return feats.flatten()
    except Exception as e:
        print("Skipping:", img_path, "->", e)
        return None

def run(input_dir, output_csv, model_name='resnet50'):
    base_model, preprocess_fn, input_size = get_model_and_preprocess(model_name)
    rows = []

    labels = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    print("Found labels:", labels)

    for lbl in labels:
        folder = os.path.join(input_dir, lbl)
        for fname in tqdm(os.listdir(folder), desc=f"Processing {lbl}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(folder, fname)
            feats = extract_one(base_model, preprocess_fn, input_size, path)
            if feats is not None:
                rows.append([fname, lbl] + feats.tolist())

    if not rows:
        raise RuntimeError("No features extracted!")

    columns = ['image_name', 'label'] + [f'f{i}' for i in range(1, len(rows[0]) - 1)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print("Saved:", output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing real/ and fake/")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model", default="resnet50")
    args = parser.parse_args()

    run(args.input_dir, args.output_csv, args.model)
