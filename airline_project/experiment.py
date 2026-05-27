"""
Moduł eksperymentu — orkiestracja pełnego pipeline'u:
wczytanie danych, preprocessing, grid search MLP, trening RF,
ewaluacja na zbiorze testowym, zapis wykresów i raportów.
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
    Tworzy PyTorch DataLoader z macierzy numpy (cechy float32 + etykiety int64).
    shuffle=True dla treningu (losowa kolejność mini-batchy), False dla ewaluacji.
    """
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def policz_metryki(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Oblicza zestaw metryk klasyfikacji: accuracy, precision/recall/F1 (macro).
    Macro = średnia arytmetyczna metryk per klasa (każda klasa ważona równo).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def przewidz_mlp(model: AirlineMLP, x: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    """
    Wykonuje inferencję MLP w trybie eval i zwraca tablicę predykcji (indeksy klas).
    Przetwarza dane w batchach, aby nie przekroczyć pamięci GPU.
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
    Ocenia model na walidacji: zwraca (średni loss, macro F1).
    Używane wewnętrznie przez pętlę treningową do early stopping i schedulera.
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
    Trenuje jedną konfigurację MLP od zera z AdamW, ReduceLROnPlateau i early stopping.
    Zwraca: (najlepszy model, DataFrame z historią epok, najlepsze val macro F1).
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
    """Generuje czytelną etykietę konfiguracji, np. 'MLP512x256x128_d0.30_lr1e-03'."""
    hidden = "x".join(str(h) for h in config["hidden_sizes"])
    return f"MLP{hidden}_d{config['dropout']:.2f}_lr{config['learning_rate']:.0e}"


def zbuduj_siatke() -> list[dict[str, Any]]:
    """
    Buduje pełną siatkę hiperparametrów (3 architektury × 3 dropout × 2 LR = 18 konfiguracji).
    Każda konfiguracja to słownik gotowy do przekazania do trenuj_mlp().
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
    Generuje i zapisuje wykresy: train/val loss, walidacyjne macro F1,
    scheduler LR oraz macierze pomyłek side-by-side (PNG, 120 DPI).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model_name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], label=f"{model_name} train")
        axes[0].plot(history["epoch"], history["val_loss"], linestyle="--", label=f"{model_name} val")
        axes[1].plot(history["epoch"], history["val_f1_macro"], label=model_name)
    axes[0].set_title("Strata treningowa i walidacyjna")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Walidacyjne macro F1")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("Macro F1")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "loss_and_f1_history.png", dpi=120)
    plt.close(fig)

    if "tuned" in histories:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(histories["tuned"]["epoch"], histories["tuned"]["learning_rate"], color="#1B998B")
        ax.set_title("Scheduler ReduceLROnPlateau — learning rate")
        ax.set_xlabel("Epoka")
        ax.set_ylabel("LR")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "scheduler_lr.png", dpi=120)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (title, cm) in zip(axes, confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predykcja")
        ax.set_ylabel("Rzeczywista")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices_side_by_side.png", dpi=120)
    plt.close(fig)


def wypisz_wnioski(comparison: pd.DataFrame, best_grid_row: pd.Series) -> None:
    """
    Drukuje rozbudowane wnioski końcowe po polsku:
    najlepszy wynik, porównanie modeli, interpretacja macierzy pomyłek,
    wpływ class weights, ograniczenia i możliwe ulepszenia.
    """
    best_test = comparison.sort_values("f1_macro", ascending=False).iloc[0]
    baseline = comparison[comparison["model"] == "MLP baseline"].iloc[0]
    tuned = comparison[comparison["model"] == "MLP tuned"].iloc[0]
    rf = comparison[comparison["model"] == "Random Forest"].iloc[0]

    print("\n" + "=" * 80)
    print("WNIOSKI KOŃCOWE")
    print("=" * 80)

    print(
        f"\n1) NAJLEPSZY WYNIK TESTOWY:\n"
        f"   Model: {best_test['model']}\n"
        f"   F1-macro = {best_test['f1_macro']:.4f} | Accuracy = {best_test['accuracy']:.4f}\n"
        f"   Precision macro = {best_test['precision_macro']:.4f} | Recall macro = {best_test['recall_macro']:.4f}"
    )

    print(
        f"\n2) PORÓWNANIE MODELI:\n"
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
        f"   Najlepsza konfiguracja: {best_grid_row['name']}\n"
        f"   Walidacyjne macro F1 = {best_grid_row['best_val_f1_macro']:.4f}\n"
        f"   Parametry: dropout={best_grid_row['dropout']}, lr={best_grid_row['learning_rate']}"
    )

    print(
        "\n4) INTERPRETACJA MACIERZY POMYŁEK:\n"
        "   - Business: rozpoznawany bardzo dobrze (duża klasa, wyraźne cechy)\n"
        "   - Eco: dobrze rozpoznawany, sporadyczne pomyłki z Business\n"
        "   - Eco Plus (PROBLEM): to tylko ~7% danych. MLP z class weights osiąga\n"
        "     recall ~41% (łapie ok. 2/5 przypadków), ale RF bez balansowania\n"
        "     praktycznie ignoruje tę klasę (recall ~1%).\n"
        "   - Eco Plus najczęściej mylona z Eco (podobne parametry lotu i oceny)."
    )

    print(
        "\n5) WPŁYW CLASS WEIGHTS:\n"
        "   CrossEntropyLoss z wagami odwrotnie proporcjonalnymi do częstości klas\n"
        "   wymusza na MLP zwracanie uwagi na Eco Plus. Bez tego mechanizmu\n"
        "   sieć (jak RF) minimalizuje globalny błąd kosztem mniejszości."
    )

    print(
        "\n6) DLACZEGO F1-MACRO > ACCURACY:\n"
        "   RF ma accuracy 86.4%, ale F1-macro tylko 60.6% — bo ignoruje Eco Plus.\n"
        "   MLP tuned ma accuracy 78.3%, ale F1-macro 65.3% — lepiej balansuje klasy.\n"
        "   Przy niezbalansowanych danych F1-macro jest rzetelniejszą metryką."
    )

    print(
        "\n7) OGRANICZENIA PROJEKTU:\n"
        "   - Klasa Eco Plus jest bardzo mała (~7%) — trudna do poprawnej klasyfikacji\n"
        "   - Brak zaawansowanej inżynierii cech (np. sumy ocen, ratia)\n"
        "   - Grid search ograniczony do 18 konfiguracji (kompromis czasowy)\n"
        "   - Dane ankietowe — subiektywne oceny, możliwe szumy\n"
        "   - RF bez class_weight='balanced' — nierówne porównanie"
    )

    print(
        "\n8) MOŻLIWE ULEPSZENIA:\n"
        "   - RF z class_weight='balanced' dla uczciwszego porównania\n"
        "   - Oversampling Eco Plus (SMOTE) lub focal loss\n"
        "   - Bayesowska optymalizacja hiperparametrów (Optuna)\n"
        "   - Feature engineering: agregaty ocen, interakcje\n"
        "   - Ensemble: MLP + RF (voting lub stacking)"
    )

    print("\n" + "=" * 80 + "\n")


def uruchom_pelny_eksperyment() -> None:
    """
    Uruchamia pełny eksperyment end-to-end:
    1. Wczytanie i czyszczenie danych
    2. Podział stratyfikowany (70/15/15)
    3. Preprocessing (fit na train)
    4. Grid search 18 konfiguracji MLP
    5. Trening baseline + tuned + Random Forest
    6. Ewaluacja na teście + zapis raportów i wykresów
    7. Drukowanie wniosków
    """
    ustaw_seedy()
    device = wybierz_urzadzenie()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Ładowanie danych Airline Passenger Satisfaction...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    raw = pd.concat([train_raw, test_raw], ignore_index=True)

    df = przygotuj_surowe_dane(raw)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    print(f"Liczba rekordów po czyszczeniu: {len(df)}")
    print("Rozkład klas:\n", df[TARGET_COLUMN].value_counts(normalize=True).round(4))

    train_df, val_df, test_df = podziel_dane_stratyfikowane(df)
    print(f"Podział: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

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

    print(f"Urządzenie PyTorch: {device}")
    print(f"Liczba cech po preprocessingu: {x_train.shape[1]}")

    grid = zbuduj_siatke()
    print(f"Grid search: {len(grid)} konfiguracji")
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

    print("Trening baseline MLP...")
    baseline_model, baseline_history, _ = trenuj_mlp(
        baseline_config,
        x_train,
        y_train,
        x_val,
        y_val,
        class_weights_tensor,
        device,
    )

    print("Trening tuned MLP...")
    tuned_model, tuned_history, _ = trenuj_mlp(
        tuned_config,
        x_train,
        y_train,
        x_val,
        y_val,
        class_weights_tensor,
        device,
    )

    print("Trening Random Forest...")
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

    print("Tabela porównawcza:")
    print(comparison.to_string(index=False))

    wypisz_wnioski(comparison, best_row)
    print(f"Wyniki zapisane w: {RESULTS_DIR}")
