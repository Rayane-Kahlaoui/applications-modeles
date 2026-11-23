"""Compare frozen pretrained CNN features on the Severstal image-alert task."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import joblib
import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torchvision.models import efficientnet_b0, efficientnet_b2, mobilenet_v3_large, mobilenet_v3_small

from projet.mlops.tracking import start_run
from projet.preprocessing.severstal import SeverstalDataset
from projet.training.baselines import choose_threshold
from projet.training.tune_binary import metrics_for
from projet.timing import benchmark_inference


MODELS = {"mobilenet-v3-small": (mobilenet_v3_small, "mobilenet_v3_small-047dcff4.pth"),
          "mobilenet-v3-large": (mobilenet_v3_large, "mobilenet_v3_large-5c1a4163.pth"),
          "efficientnet-b0": (efficientnet_b0, "efficientnet_b0_rwightman-7f5810bc.pth"),
          "efficientnet-b2": (efficientnet_b2, "efficientnet_b2_rwightman-c35c1473.pth")}


class CustomCNN(nn.Module):
    """Compact image classifier initialized entirely from scratch."""

    def __init__(self):
        super().__init__()
        layers = []
        for input_channels, output_channels in ((1, 16), (16, 32), (32, 64), (64, 128)):
            layers.extend([nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                           nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, 1))

    def forward(self, images):
        features = self.features(images)
        pooled = torch.cat((nn.functional.adaptive_avg_pool2d(features, 1).flatten(1),
                            nn.functional.adaptive_max_pool2d(features, 1).flatten(1)), dim=1)
        return self.head(pooled).squeeze(1)


class ImageSamples(Dataset):
    """Cache resized images and preserve image-level labels during augmentation."""

    def __init__(self, images, masks, labels, indices, *, augment=False, seed=0):
        self.images, self.masks, self.labels, self.indices = images, masks, labels, indices
        self.augment, self.seed, self.epoch = augment, seed, 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        image = self.images[source_index]
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, source_index]))
            angle = float(rng.uniform(-2, 2))
            matrix = cv2.getRotationMatrix2D((127.5, 31.5), angle, 1)
            rotated = cv2.warpAffine(image, matrix, (256, 64), borderMode=cv2.BORDER_REFLECT_101)
            mask = self.masks[source_index]
            transformed = cv2.warpAffine(mask, matrix, (256, 64), flags=cv2.INTER_NEAREST)
            if not self.labels[source_index] or (mask.any() and transformed.any()):
                image = rotated
            if rng.random() < 0.5:
                image = cv2.flip(image, 1)
            if rng.random() < 0.25:
                image = cv2.GaussianBlur(image, (3, 3), 0.7)
            image = np.clip(image.astype(np.float32) * rng.uniform(0.85, 1.15)
                            + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
        return torch.from_numpy(image.copy()).unsqueeze(0).float() / 255, float(self.labels[source_index])


def cache_images(dataset, indices):
    images, masks, labels, ids, classes = [], [], [], [], []
    for number, index in enumerate(indices, start=1):
        sample = dataset[int(index)]
        images.append(cv2.resize(sample["image"], (256, 64), interpolation=cv2.INTER_AREA))
        mask = np.any(sample["masks"], axis=0).astype(np.uint8)
        masks.append(cv2.resize(mask, (256, 64), interpolation=cv2.INTER_NEAREST))
        labels.append(int(bool(len(sample["labels"]))))
        ids.append(sample["image_id"])
        classes.append([int(class_id in sample["labels"]) for class_id in range(4)])
        if number % 1000 == 0:
            print(f"cached {number}/{len(indices)} images", flush=True)
    return np.stack(images), np.stack(masks), np.asarray(labels), ids, np.asarray(classes)


def run_custom(output, seed, train_count, validation_count, batch_size, epochs):
    torch.manual_seed(seed)
    train, validation = SeverstalDataset("train"), SeverstalDataset("validation")
    train_indices = np.random.default_rng(seed).permutation(len(train))[:train_count]
    valid_indices = np.random.default_rng(seed).permutation(len(validation))[:validation_count]
    with start_run("custom-cnn-scratch", phase="binary-training", seed=seed,
                   description="Scratch CNN with training-only holdout and augmentation.") as run:
        mlflow.log_params({"train_count": len(train_indices), "validation_count": len(valid_indices),
            "image_size": "256x64", "batch_size": batch_size, "max_epochs": epochs,
            "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
            "augmentation": "flip, rotation +/-2 degrees, brightness 0.85-1.15, noise std 4, optional blur",
            "checkpoint_selection": "training holdout average precision", "training_holdout_fraction": 0.1,
            "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
            "manifest_sha256": hashlib.sha256((train.root / "split.json").read_bytes()).hexdigest()})
        start = time.perf_counter()
        train_images, train_masks, train_labels, train_ids, _ = cache_images(train, train_indices)
        valid_images, valid_masks, valid_labels, valid_ids, classes = cache_images(validation, valid_indices)
        cache_seconds = time.perf_counter() - start
        fit, holdout = train_test_split(np.arange(len(train_labels)), test_size=0.1, random_state=seed, stratify=train_labels)
        tune, comparison = train_test_split(np.arange(len(valid_labels)), test_size=0.5, random_state=seed, stratify=valid_labels)
        fitting = ImageSamples(train_images, train_masks, train_labels, fit, augment=True, seed=seed)
        held = ImageSamples(train_images, train_masks, train_labels, holdout)
        generator = torch.Generator().manual_seed(seed)
        fit_loader = DataLoader(fitting, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
        holdout_loader = DataLoader(held, batch_size=batch_size, shuffle=False, num_workers=0)
        model = CustomCNN()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_function = nn.BCEWithLogitsLoss()
        best_score, best_state, best_epoch = -1, None, 0
        start = time.perf_counter()
        for epoch in range(epochs):
            fitting.epoch = epoch
            model.train()
            losses = []
            for images, labels in fit_loader:
                optimizer.zero_grad()
                loss = loss_function(model(images), labels.float())
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            holdout_scores, holdout_labels = [], []
            with torch.inference_mode():
                for images, labels in holdout_loader:
                    holdout_scores.extend(torch.sigmoid(model(images)).tolist())
                    holdout_labels.extend(labels.tolist())
            ap = float(average_precision_score(holdout_labels, holdout_scores))
            mlflow.log_metrics({"training_loss": float(np.mean(losses)), "training_holdout_ap": ap}, step=epoch)
            print(f"custom CNN epoch {epoch + 1}/{epochs}: training holdout AP={ap:.4f}", flush=True)
            if ap > best_score:
                best_score, best_epoch = ap, epoch + 1
                best_state = {key: value.clone() for key, value in model.state_dict().items()}
            if epoch + 1 - best_epoch >= 3:
                break
        fit_seconds = time.perf_counter() - start
        model.load_state_dict(best_state)
        model.eval()
        valid_loader = DataLoader(ImageSamples(valid_images, valid_masks, valid_labels, np.arange(len(valid_labels))),
                                  batch_size=batch_size, shuffle=False, num_workers=0)
        scores = []
        with torch.inference_mode():
            for images, _ in valid_loader:
                scores.extend(torch.sigmoid(model(images)).tolist())
        scores = np.asarray(scores)
        threshold = choose_threshold(valid_labels[tune], scores[tune])
        tune_metrics = metrics_for(valid_labels[tune], scores[tune], threshold)
        comparison_metrics = metrics_for(valid_labels[comparison], scores[comparison], threshold)
        mlflow.log_metrics({f"tune_{key}": value for key, value in tune_metrics.items() if value is not None})
        mlflow.log_metrics({f"comparison_{key}": value for key, value in comparison_metrics.items() if value is not None})
        mlflow.log_metrics({"best_training_holdout_ap": best_score, "best_epoch": best_epoch,
                            "training_seconds": fit_seconds, "cache_seconds": cache_seconds})
        stress_results = {}
        for scenario in ("darker", "brighter", "noise", "blur"):
            rng = np.random.default_rng(seed)
            stress_scores = []
            with torch.inference_mode():
                for offset in range(0, len(comparison), batch_size):
                    altered = []
                    for index in comparison[offset:offset + batch_size]:
                        image = valid_images[index].astype(np.float32)
                        if scenario == "darker":
                            image *= 0.75
                        elif scenario == "brighter":
                            image *= 1.25
                        elif scenario == "noise":
                            image += rng.normal(0, 8, image.shape)
                        else:
                            image = cv2.GaussianBlur(image, (0, 0), 1)
                        altered.append(np.clip(image, 0, 255).astype(np.uint8))
                    tensor = torch.from_numpy(np.stack(altered)).unsqueeze(1).float() / 255
                    stress_scores.extend(torch.sigmoid(model(tensor)).tolist())
            stress = metrics_for(valid_labels[comparison], np.asarray(stress_scores), threshold)
            stress_results[scenario] = stress
            mlflow.log_metrics({f"stress_{scenario}_{key}": value for key, value in stress.items()
                                if value is not None})
        class_results = {}
        for class_id in range(4):
            present = classes[comparison, class_id].astype(bool)
            support = int(present.sum())
            recall = float(np.mean(scores[comparison][present] >= threshold)) if support else None
            class_results[str(class_id + 1)] = {"support": support, "image_alert_recall": recall}
            if recall is not None:
                mlflow.log_metric(f"comparison_class_{class_id + 1}_image_alert_recall", recall)
        output.mkdir(parents=True)
        torch.save(model.state_dict(), output / "model.pt")
        result = {"run_id": run.info.run_id, "model": "custom-cnn-scratch", "train_ids": train_ids,
                  "fit_ids": [train_ids[i] for i in fit], "training_holdout_ids": [train_ids[i] for i in holdout],
                  "validation_ids": valid_ids, "tune_ids": [valid_ids[i] for i in tune],
                  "comparison_ids": [valid_ids[i] for i in comparison], "validation_labels": valid_labels.tolist(),
                  "validation_scores": scores.tolist(), "tune_metrics": tune_metrics,
                  "comparison_metrics": comparison_metrics, "comparison_class_image_alert_recall": class_results,
                  "stress_comparison_metrics": stress_results,
                  "best_epoch": best_epoch, "training_seconds": fit_seconds, "cache_seconds": cache_seconds}
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifacts(str(output))
        mlflow.log_artifact(__file__, artifact_path="source")
        print(f"custom CNN: tuning={tune_metrics}, comparison={comparison_metrics}, class2={class_results['2']}", flush=True)


def extract(dataset, indices, model, batch_size):
    vectors, labels, ids, classes = [], [], [], []
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(indices), batch_size):
            batch = []
            for index in indices[offset:offset + batch_size]:
                sample = dataset[int(index)]
                image = cv2.resize(sample["image"], (256, 64), interpolation=cv2.INTER_AREA)
                batch.append(image)
                labels.append(int(bool(len(sample["labels"]))))
                classes.append([int(class_id in sample["labels"]) for class_id in range(4)])
                ids.append(sample["image_id"])
            tensor = torch.from_numpy(np.stack(batch)).unsqueeze(1).repeat(1, 3, 1, 1).float() / 255
            features = model.avgpool(model.features((tensor - mean) / std)).flatten(1)
            vectors.append(features.cpu().numpy())
            if (offset // batch_size) % 20 == 0:
                print(f"features: {min(offset + batch_size, len(indices))}/{len(indices)}", flush=True)
    return np.concatenate(vectors), np.asarray(labels), ids, np.asarray(classes), time.perf_counter() - start


def run(name, weights_dir, output, seed, train_count, validation_count, batch_size):
    train, validation = SeverstalDataset("train"), SeverstalDataset("validation")
    train_indices = np.random.default_rng(seed).permutation(len(train))[:train_count]
    validation_indices = np.random.default_rng(seed).permutation(len(validation))[:validation_count]
    factory, filename = MODELS[name]
    weights = weights_dir / filename
    if not weights.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights}")
    with start_run(name, phase="binary-frozen-features", seed=seed,
                   description="Image-level frozen CNN feature extraction; separate validation tuning and comparison.") as run:
        model = factory(weights=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.eval()
        mlflow.log_params({"weights_file": filename, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                           "backbone": "frozen ImageNet-1k", "image_size": "256x64",
                           "preprocessing": "grayscale replicated to RGB; ImageNet normalization",
                           "head": "StandardScaler + balanced LogisticRegression C=1", "batch_size": batch_size,
                           "train_count": len(train_indices), "validation_count": len(validation_indices),
                           "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
                           "torch_threads": torch.get_num_threads(),
                           "manifest_sha256": hashlib.sha256((train.root / "split.json").read_bytes()).hexdigest()})
        train_x, train_y, train_ids, _, train_seconds = extract(train, train_indices, model, batch_size)
        valid_x, valid_y, valid_ids, classes, valid_seconds = extract(validation, validation_indices, model, batch_size)
        if len(np.unique(train_y)) != 2 or len(np.unique(valid_y)) != 2:
            raise ValueError("Both splits must contain positive and negative images")
        tune, comparison = train_test_split(np.arange(len(valid_y)), test_size=0.5, random_state=seed, stratify=valid_y)
        head = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight="balanced", max_iter=1000, random_state=seed))
        start = time.perf_counter()
        head.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - start
        scores = head.predict_proba(valid_x)[:, 1]
        threshold = choose_threshold(valid_y[tune], scores[tune])
        tune_metrics = metrics_for(valid_y[tune], scores[tune], threshold)
        comparison_metrics = metrics_for(valid_y[comparison], scores[comparison], threshold)
        mlflow.log_metrics({f"tune_{key}": value for key, value in tune_metrics.items() if value is not None})
        mlflow.log_metrics({f"comparison_{key}": value for key, value in comparison_metrics.items() if value is not None})
        mlflow.log_metrics({"head_fit_seconds": fit_seconds, "train_extraction_seconds": train_seconds,
                            "validation_extraction_seconds": valid_seconds})
        class_results = {}
        for class_id in range(4):
            present = classes[comparison, class_id].astype(bool)
            support = int(present.sum())
            recall = float(np.mean(scores[comparison][present] >= threshold)) if support else None
            class_results[str(class_id + 1)] = {"support": support, "image_alert_recall": recall}
            if recall is not None:
                mlflow.log_metric(f"comparison_class_{class_id + 1}_image_alert_recall", recall)
        output.mkdir(parents=True)
        joblib.dump(head, output / "head.joblib")
        result = {"run_id": run.info.run_id, "model": name, "weights": filename,
                  "train_ids": train_ids, "validation_ids": valid_ids,
                  "tune_ids": [valid_ids[i] for i in tune], "comparison_ids": [valid_ids[i] for i in comparison],
                  "validation_labels": valid_y.tolist(), "validation_scores": scores.tolist(),
                  "tune_metrics": tune_metrics, "comparison_metrics": comparison_metrics,
                  "comparison_class_image_alert_recall": class_results,
                  "train_extraction_seconds": train_seconds, "validation_extraction_seconds": valid_seconds,
                  "head_fit_seconds": fit_seconds}
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifacts(str(output))
        mlflow.log_artifact(str(weights), artifact_path="backbone")
        mlflow.log_artifact(__file__, artifact_path="source")
        print(f"{name}: tuning={tune_metrics}, comparison={comparison_metrics}, class2={class_results['2']}", flush=True)


def benchmark(name, weights_dir, result_dir, count, threads):
    """Record decoded-image-to-score latency on a shared comparison subset."""
    from projet.mlops.tracking import configure_tracking
    from projet.training.baselines import extract_features

    result = json.loads((result_dir / "results.json").read_text(encoding="utf-8"))
    ids = result["comparison_ids"][:count]
    if not ids:
        raise ValueError("No comparison images available for timing")
    root = SeverstalDataset("validation").root
    images = []
    for image_id in ids:
        image = cv2.imread(str(root / "images" / image_id), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_id)
        images.append(image)
    if name == "custom-cnn":
        model = CustomCNN()
        model.load_state_dict(torch.load(result_dir / "model.pt", map_location="cpu", weights_only=True))
        model.eval()

        def predict(image):
            resized = cv2.resize(image, (256, 64), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(resized).reshape(1, 1, 64, 256).float() / 255
            with torch.inference_mode():
                return torch.sigmoid(model(tensor))
    elif name == "random-forest":
        head = joblib.load(result_dir / "model.joblib")
        head.set_params(n_jobs=threads)

        def predict(image):
            return head.predict_proba(extract_features(image, "texture-forest").reshape(1, -1))
    else:
        head = joblib.load(result_dir / "head.joblib")
        factory, filename = MODELS[name]
        model = factory(weights=None)
        model.load_state_dict(torch.load(weights_dir / filename, map_location="cpu", weights_only=True))
        model.eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        def predict(image):
            resized = cv2.resize(image, (256, 64), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(resized).reshape(1, 1, 64, 256).repeat(1, 3, 1, 1).float() / 255
            with torch.inference_mode():
                features = model.avgpool(model.features((tensor - mean) / std)).flatten(1).numpy()
            return head.predict_proba(features)
    cv2.setNumThreads(threads)
    with threadpool_limits(limits=threads):
        timing = benchmark_inference(predict, images, warmup=5)
    timing.update({"sample_ids": ids, "threads": threads, "cpu": platform.processor(),
                   "boundary": "decoded grayscale image through preprocessing, model and score; disk excluded"})
    path = result_dir / "timing.json"
    if path.exists():
        raise FileExistsError(f"Timing result already exists: {path}")
    path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    configure_tracking()
    with mlflow.start_run(run_id=result["run_id"]):
        mlflow.log_metrics({f"benchmark_{key}": timing[key] for key in
                            ("mean_seconds", "median_seconds", "p95_seconds", "max_seconds")})
        mlflow.log_param("benchmark_threads", threads)
        mlflow.log_artifact(str(path))
        mlflow.log_artifact(__file__, artifact_path="benchmark_source")
    print(f"{name}: median={timing['median_seconds']:.4f}s p95={timing['p95_seconds']:.4f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "random-forest", "custom-cnn"), required=True)
    parser.add_argument("--weights-dir", type=Path, default=Path(".cache/weights"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=10050)
    parser.add_argument("--validation-count", type=int, default=1252)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if min(args.train_count, args.validation_count, args.batch_size, args.threads, args.epochs) < 1 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative")
    torch.set_num_threads(args.threads)
    if args.benchmark_only:
        benchmark(args.model, args.weights_dir, args.output, min(args.validation_count, 100), args.threads)
    else:
        if args.output.exists() or args.model == "random-forest":
            parser.error("Training requires a new output directory and a CNN model")
        if args.model == "custom-cnn":
            run_custom(args.output, args.seed, args.train_count, args.validation_count, args.batch_size, args.epochs)
        else:
            run(args.model, args.weights_dir, args.output, args.seed, args.train_count, args.validation_count, args.batch_size)


if __name__ == "__main__":
    main()
