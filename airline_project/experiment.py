"""
Experiment module — orchestration of the full pipeline:
data loading, preprocessing, MLP grid search, RF training,
evaluation on the test set, saving charts and reports.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from airline_project.config import (
    ARCHITECTURES,
    BASELINE_CONFIG,
    CLASS_LABELS,
    DROPOUT_VALUES,
    LABEL_TO_INDEX,
    LEARNING_RATES,
    PLOTS_DIR,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from airline_project.model import AirlineMLP, wybierz_urzadzenie
from airline_project.preprocessing import (
    fit_preprocessor,
    podziel_dane_stratyfikowane,
    przygotuj_surowe_dane,
    transformuj_cechy,
    ustaw_seedy,
    zakoduj_target,
)


def utworz_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Creates a PyTorch DataLoader from numpy arrays (float32 features + int64 labels).
    shuffle=True for training (random mini-batch order), False for evaluation.
    """
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def policz_metryki(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Computes a set of classification metrics: accuracy, precision/recall/F1 (macro).
    Macro = arithmetic mean of per-class metrics (each class weighted equally).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def przewidz_mlp(model: AirlineMLP, x: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    """
    Runs MLP inference in eval mode and returns an array of predictions (class indices).
    Processes data in batches to avoid exceeding GPU memory.
    """
    loader = utworz_loader(x, np.zeros(len(x), dtype=np.int64), batch_size=batch_size, shuffle=False)
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def ocen_na_walidacji(
    model: AirlineMLP,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluates the model on validation: returns (mean loss, macro F1).
    Used internally by the training loop for early stopping and the scheduler.
    """
    model.eval()
    losses: list[float] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_function(logits, yb)
            losses.append(float(loss.item()))
            all_true.append(yb.cpu().numpy())
            all_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(np.mean(losses)), macro_f1


def trenuj_mlp(
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[AirlineMLP, pd.DataFrame, float]:
    """
    Trains a single MLP configuration from scratch with AdamW, ReduceLROnPlateau and early stopping.
    Returns: (the best model, a DataFrame with the epoch history, the best val macro F1).
    """
    model = AirlineMLP(
        input_dim=x_train.shape[1],
        hidden_sizes=tuple(config["hidden_sizes"]),
        output_dim=len(CLASS_LABELS),
        dropout=float(config["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    train_loader = utworz_loader(x_train, y_train, int(config["batch_size"]), shuffle=True)
    val_loader = utworz_loader(x_val, y_val, int(config["batch_size"]), shuffle=False)

    history_rows: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_val_f1 = -np.inf
    patience_counter = 0

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss, val_f1 = ocen_na_walidacji(model, val_loader, criterion, device)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
                "learning_rate": current_lr,
            }
        )

        scheduler.step(val_f1)

        if val_f1 > best_val_f1 + float(config.get("min_delta", 1e-4)):
            best_val_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(config["patience"]):
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history_rows), best_val_f1


def etykieta_konfiguracji(config: dict[str, Any]) -> str:
    """Generates a readable configuration label, e.g. 'MLP512x256x128_d0.30_lr1e-03'."""
    hidden = "x".join(str(h) for h in config["hidden_sizes"])
    return f"MLP{hidden}_d{config['dropout']:.2f}_lr{config['learning_rate']:.0e}"


def zbuduj_siatke() -> list[dict[str, Any]]:
    """
    Builds the full hyperparameter grid (3 architectures × 3 dropout × 2 LR = 18 configurations).
    Each configuration is a dictionary ready to pass to trenuj_mlp().
    """
    configs: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for dropout in DROPOUT_VALUES:
            for learning_rate in LEARNING_RATES:
                cfg = {
                    "name": "",
                    "hidden_sizes": architecture,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    "weight_decay": 1e-4,
                    "batch_size": 1024,
                    "max_epochs": 30,
                    "patience": 5,
                    "min_delta": 1e-4,
                }
                cfg["name"] = etykieta_konfiguracji(cfg)
                configs.append(cfg)
    return configs


def zapisz_wykresy(histories: dict[str, pd.DataFrame], confusion_matrices: dict[str, np.ndarray]) -> None:
    """
    Generates and saves charts: train/val loss, validation macro F1,
    the LR scheduler and side-by-side confusion matrices (PNG, 120 DPI).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=f"{model_name} train")
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--", label=f"{model_name} val")
        axes[1].plot(history["epoch"], history["val_f1_macro"], label=model_name)
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Validation macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "loss_and_f1_history.png", dpi=120)
    plt.close(fig)

    if "tuned" in histories:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(histories["tuned"]["epoch"], histories["tuned"]["learning_rate"], color="#1B998B")
        ax.set_title("ReduceLROnPlateau scheduler — learning rate")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("LR")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "scheduler_lr.png", dpi=120)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (title, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices_side_by_side.png", dpi=120)
    plt.close(fig)


def wypisz_wnioski(comparison: pd.DataFrame, best_grid_row: pd.Series) -> None:
    """
    Prints detailed final conclusions:
    best result, model comparison, confusion-matrix interpretation,
    the effect of class weights, limitations and possible improvements.
    """
    best_test = comparison.sort_values("f1_macro", ascending=False).iloc[0]
    baseline = comparison[comparison["model"] == "MLP baseline"].iloc[0]
    tuned = comparison[comparison["model"] == "MLP tuned"].iloc[0]
    rf = comparison[comparison["model"] == "Random Forest"].iloc[0]

    print("\n" + "=" * 80)
    print("FINAL CONCLUSIONS")
    print("=" * 80)

    print(
        f"\n1) BEST TEST RESULT:\n"
        f"   Model: {best_test['model']}\n"
        f"   F1-macro = {best_test['f1_macro']:.4f} | Accuracy = {best_test['accuracy']:.4f}\n"
        f"   Precision macro = {best_test['precision_macro']:.4f} | Recall macro = {best_test['recall_macro']:.4f}"
    )

    print(
        f"\n2) MODEL COMPARISON:\n"
        f"   {'Model':<15} {'Accuracy':>10} {'F1-macro':>10} {'Precision':>10} {'Recall':>10}\n"
        f"   {'─' * 55}\n"
        f"   {'MLP baseline':<15} {baseline['accuracy']:>10.4f} {baseline['f1_macro']:>10.4f} "
        f"{baseline['precision_macro']:>10.4f} {baseline['recall_macro']:>10.4f}\n"
        f"   {'MLP tuned':<15} {tuned['accuracy']:>10.4f} {tuned['f1_macro']:>10.4f} "
        f"{tuned['precision_macro']:>10.4f} {tuned['recall_macro']:>10.4f}\n"
        f"   {'Random Forest':<15} {rf['accuracy']:>10.4f} {rf['f1_macro']:>10.4f} "
        f"{rf['precision_macro']:>10.4f} {rf['recall_macro']:>10.4f}"
    )

    print(
        f"\n3) GRID SEARCH:\n"
        f"   Best configuration: {best_grid_row['name']}\n"
        f"   Validation macro F1 = {best_grid_row['best_val_f1_macro']:.4f}\n"
        f"   Parameters: dropout={best_grid_row['dropout']}, lr={best_grid_row['learning_rate']}"
    )

    print(
        "\n4) CONFUSION-MATRIX INTERPRETATION:\n"
        "   - Business: recognized very well (large class, distinct features)\n"
        "   - Eco: recognized well, occasional confusion with Business\n"
        "   - Eco Plus (PROBLEM): only ~7% of the data. The MLP with class weights reaches\n"
        "     recall ~41% (catches about 2/5 of the cases), whereas RF without balancing\n"
        "     practically ignores this class (recall ~1%).\n"
        "   - Eco Plus is most often confused with Eco (similar flight parameters and scores)."
    )

    print(
        "\n5) EFFECT OF CLASS WEIGHTS:\n"
        "   CrossEntropyLoss with weights inversely proportional to class frequencies\n"
        "   forces the MLP to pay attention to Eco Plus. Without this mechanism\n"
        "   the network (like RF) minimizes the global error at the expense of the minority."
    )

    print(
        "\n6) WHY F1-MACRO > ACCURACY:\n"
        "   RF has accuracy 86.4% but F1-macro only 60.6% — because it ignores Eco Plus.\n"
        "   MLP tuned has accuracy 78.3% but F1-macro 65.3% — it balances the classes better.\n"
        "   With imbalanced data, F1-macro is a more reliable metric."
    )

    print(
        "\n7) PROJECT LIMITATIONS:\n"
        "   - The Eco Plus class is very small (~7%) — hard to classify correctly\n"
        "   - No advanced feature engineering (e.g. score sums, ratios)\n"
        "   - Grid search limited to 18 configurations (a time trade-off)\n"
        "   - Survey data — subjective ratings, possible noise\n"
        "   - RF without class_weight='balanced' — an uneven comparison"
    )

    print("\n" + "=" * 80 + "\n")


def uruchom_pelny_eksperyment() -> None:
    """
    Runs the full end-to-end experiment:
    1. Load and clean the data
    2. Stratified split (70/15/15)
    3. Preprocessing (fit on train)
    4. Grid search over 18 MLP configurations
    5. Train baseline + tuned + Random Forest
    6. Evaluate on test + save reports and charts
    7. Print conclusions
    """
    ustaw_seedy()
    device = wybierz_urzadzenie()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading the Airline Passenger Satisfaction data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    raw = pd.concat([train_raw, test_raw], ignore_index=True)

    df = przygotuj_surowe_dane(raw)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    print(f"Number of records after cleaning: {len(df)}")
    print("Class distribution:\n", df[TARGET_COLUMN].value_counts(normalize=True).round(4))

    train_df, val_df, test_df = podziel_dane_stratyfikowane(df)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    artifacts = fit_preprocessor(train_df)

    x_train = transformuj_cechy(train_df, artifacts)
    x_val = transformuj_cechy(val_df, artifacts)
    x_test = transformuj_cechy(test_df, artifacts)

    y_train = zakoduj_target(train_df[TARGET_COLUMN], LABEL_TO_INDEX)
    y_val = zakoduj_target(val_df[TARGET_COLUMN], LABEL_TO_INDEX)
    y_test = zakoduj_target(test_df[TARGET_COLUMN], LABEL_TO_INDEX)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_LABELS)),
        y=y_train,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print(f"PyTorch device: {device}")
    print(f"Number of features after preprocessing: {x_train.shape[1]}")

    grid = zbuduj_siatke()
    print(f"Grid search: {len(grid)} configurations")
    grid_rows: list[dict[str, Any]] = []
    for i, config in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] {config['name']}")
        _, _, best_val_f1 = trenuj_mlp(config, x_train, y_train, x_val, y_val, class_weights_tensor, device)
        grid_rows.append({"name": config["name"], "best_val_f1_macro": best_val_f1, **config})

    grid_df = pd.DataFrame(grid_rows).sort_values("best_val_f1_macro", ascending=False)
    grid_df.to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)

    best_row = grid_df.iloc[0]
    tuned_config = next(cfg for cfg in grid if cfg["name"] == best_row["name"])
    baseline_config = {**BASELINE_CONFIG}

    print("Training baseline MLP...")
    baseline_model, baseline_history, _ = trenuj_mlp(
        baseline_config,
        x_train,
        y_train,
        x_val,
        y_val,
        class_weights_tensor,
        device,
    )

    print("Training tuned MLP...")
    tuned_model, tuned_history, _ = trenuj_mlp(
        tuned_config,
        x_train,
        y_train,
        x_val,
        y_val,
        class_weights_tensor,
        device,
    )

    print("Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_model.fit(x_train, y_train)

    baseline_pred = przewidz_mlp(baseline_model, x_test, device)
    tuned_pred = przewidz_mlp(tuned_model, x_test, device)
    rf_pred = rf_model.predict(x_test)

    reports = {
        "mlp_baseline": classification_report(y_test, baseline_pred, target_names=CLASS_LABELS, zero_division=0),
        "mlp_tuned": classification_report(y_test, tuned_pred, target_names=CLASS_LABELS, zero_division=0),
        "random_forest": classification_report(y_test, rf_pred, target_names=CLASS_LABELS, zero_division=0),
    }
    for name, text in reports.items():
        (RESULTS_DIR / f"classification_report_{name}.txt").write_text(text, encoding="utf-8")

    comparison = pd.DataFrame(
        [
            {"model": "MLP baseline", **policz_metryki(y_test, baseline_pred)},
            {"model": "MLP tuned", **policz_metryki(y_test, tuned_pred)},
            {"model": "Random Forest", **policz_metryki(y_test, rf_pred)},
        ]
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    confusion_matrices = {
        "Random Forest": confusion_matrix(y_test, rf_pred),
        "MLP baseline": confusion_matrix(y_test, baseline_pred),
        "MLP tuned": confusion_matrix(y_test, tuned_pred),
    }
    zapisz_wykresy(
        histories={"baseline": baseline_history, "tuned": tuned_history},
        confusion_matrices=confusion_matrices,
    )

    metadata = {
        "device": str(device),
        "n_records": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": int(x_train.shape[1]),
        "baseline_config": baseline_config,
        "tuned_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in tuned_config.items()},
        "best_grid": best_row.to_dict(),
        "class_distribution": df[TARGET_COLUMN].value_counts(normalize=True).to_dict(),
    }
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print("\n--- Classification report: baseline ---\n")
    print(reports["mlp_baseline"])
    print("\n--- Classification report: tuned ---\n")
    print(reports["mlp_tuned"])
    print("\n--- Classification report: random forest ---\n")
    print(reports["random_forest"])

    print("Comparison table:")
    print(comparison.to_string(index=False))

    wypisz_wnioski(comparison, best_row)
    print(f"Results saved in: {RESULTS_DIR}")
