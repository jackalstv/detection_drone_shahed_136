"""
Entraînement YOLOv11 pour détection Shahed-136
Dataset: Military-Units-24v07v2023.v1i.yolov11
- Auto-détection du device (GPU si dispo, sinon CPU)
- Mode rapide CPU activable (FAST_MODE)
- Gel du backbone activable (FREEZE_BACKBONE)
"""

import os
import shutil
import yaml
import multiprocessing as mp
from typing import Optional, Tuple

from ultralytics import YOLO

# ======================
#   CONFIGURATION
# ======================

DATASET_PATH = "Military-Units-24v07v2023.v1i.yolov11"
MODEL_WEIGHTS = "yolo11n.pt"      # yolo11n/s/m/l/x
OUTPUT_MODEL_NAME = "shahed136_detector.pt"

# Hyperparams de base
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 20
SEED = 42

# Contrôles d’exécution
AUTO_START = True          # lance l'entraînement sans prompt
FORCE_CPU   = False        # force CPU même si CUDA dispo
FAST_MODE   = True         # accélère fortement l'entraînement sur CPU
FREEZE_BACKBONE = True     # gèle une partie du backbone pour aller plus vite

# ======================
#   DEVICE & RESSOURCES
# ======================

def select_device() -> Tuple[str, bool]:
    """
    Retourne (device_str, amp_ok)
    device_str: 'cpu', 'mps', '0', '0,1', ...
    amp_ok: True si AMP utile (GPU CUDA)
    """
    try:
        import torch
        if not FORCE_CPU and torch.cuda.is_available():
            return "0", True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", False
    except Exception:
        pass
    return "cpu", False

def autotune_batch_workers(base_batch: int, amp_ok: bool) -> Tuple[int, int]:
    cpu_count = max(1, mp.cpu_count())
    workers = min(8, max(2, cpu_count // 2))
    batch = base_batch if amp_ok else min(base_batch, 8)  # sur CPU, batch modeste
    return batch, workers

# ======================
#   data.yaml → correction
# ======================

def normalize_names(names):
    if not isinstance(names, list):
        return names
    norm = [str(int(n)) if isinstance(n, (int, float)) else str(n) for n in names]
    if len(norm) == 1 and norm[0] in {"1", "shahed136", "shahed-136"}:
        norm = ["military-drone"]
    return norm

def fix_data_yaml() -> Optional[str]:
    yaml_path = os.path.join(DATASET_PATH, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"❌ Fichier data.yaml introuvable: {yaml_path}")
        return None

    print("=" * 70)
    print("CORRECTION DU FICHIER data.yaml")
    print("=" * 70)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    dataset_abs = os.path.abspath(DATASET_PATH)
    data["path"]  = dataset_abs
    data["train"] = "train/images"
    data["val"]   = "valid/images"
    if os.path.exists(os.path.join(DATASET_PATH, "test/images")):
        data["test"] = "test/images"

    data["names"] = normalize_names(data.get("names", ["military-drone"]))

    yaml_corrected_path = os.path.join(DATASET_PATH, "data_corrected.yaml")
    with open(yaml_corrected_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"✓ Fichier corrigé: {yaml_corrected_path}")
    print(f"  - Path:  {data['path']}")
    print(f"  - Train: {data['train']}")
    print(f"  - Val:   {data['val']}")
    if data.get("test"):
        print(f"  - Test:  {data['test']}")
    print(f"  - Classes: {data['names']}")
    return yaml_corrected_path

# ======================
#   Vérification dataset
# ======================

def count_files(dir_path, exts):
    if not os.path.exists(dir_path):
        return 0
    return sum(1 for f in os.listdir(dir_path) if f.lower().endswith(exts) and not f.startswith("."))

def verify_split(split_name: str) -> bool:
    images_dir = os.path.join(DATASET_PATH, f"{split_name}/images")
    labels_dir = os.path.join(DATASET_PATH, f"{split_name}/labels")
    ok = True
    if not os.path.exists(images_dir):
        print(f"❌ Dossier manquant: {images_dir}"); ok = False
    if not os.path.exists(labels_dir):
        print(f"❌ Dossier manquant: {labels_dir}"); ok = False
    if not ok:
        return False

    n_img = count_files(images_dir, (".jpg", ".jpeg", ".png"))
    n_lbl = count_files(labels_dir, (".txt",))
    print(f"✓ {split_name}/images: {n_img} fichiers")
    print(f"✓ {split_name}/labels: {n_lbl} fichiers")

    if n_img and n_lbl:
        imgs = {os.path.splitext(f)[0] for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        lbls = {os.path.splitext(f)[0] for f in os.listdir(labels_dir)
                if f.lower().endswith(".txt")}
        matching = len(imgs & lbls)
        total = len(imgs)
        rate = 100.0 * matching / max(1, total)
        print(f"→ Correspondance {split_name}: {matching}/{total} ({rate:.1f}%)")
        if rate < 90.0:
            print("⚠ Attention: images sans labels correspondants")
    return True

def verify_dataset() -> bool:
    print("\n" + "=" * 70)
    print("VÉRIFICATION DU DATASET")
    print("=" * 70)
    ok_train = verify_split("train")
    ok_valid = verify_split("valid")
    return ok_train and ok_valid

# ======================
#   Entraînement
# ======================

def train_model(data_yaml_path: str) -> Optional[str]:
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DU MODÈLE YOLOv11")
    print("=" * 70)

    device_str, amp_ok = select_device()
    batch, workers = autotune_batch_workers(BATCH_SIZE, amp_ok)

    # Mode rapide CPU : tailles + epochs + aug plus légères
    train_overrides = dict(
        data=data_yaml_path,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=batch,
        imgsz=IMG_SIZE,
        device=device_str,
        project='runs/detect',
        name='shahed136_training',
        pretrained=True,
        optimizer='auto',
        amp=amp_ok,                 # False sur CPU/MPS
        cache=True,                 # mets False si RAM insuffisante
        workers=workers,
        seed=SEED,
        deterministic=True,
        exist_ok=True,
        rect=True,                  # batches rectangulaires (I/O plus efficaces)
        plots=False,                # moins d’E/S
        save_period=-1,             # n’enregistre pas chaque epoch intermédiaire
    )

    if FAST_MODE:
        train_overrides.update(
            epochs=30,
            patience=6,
            batch=min(batch, 8),
            imgsz=448,
            amp=False,          # CPU: inutile
            # Augmentations légères pour gagner du temps CPU
            mosaic=0.0,
            hsv_h=0.0, hsv_s=0.5, hsv_v=0.5,
            flipud=0.0, fliplr=0.25,
            # ema retiré (incompatible sur ta version)
        )

    if FREEZE_BACKBONE:
        # Gèle un bloc de couches du backbone (accélère et stabilise sur petits runs)
        train_overrides.update(freeze=10)

    print(f"\n→ Modèle: {MODEL_WEIGHTS}")
    print(f"→ Device: {device_str} | AMP: {'ON' if train_overrides['amp'] else 'OFF'}")
    print(f"→ Epochs: {train_overrides['epochs']} | Batch: {train_overrides['batch']} | img: {train_overrides['imgsz']}")
    print(f"→ Workers: {workers} | Rect: {train_overrides['rect']}")
    if FREEZE_BACKBONE:
        print(f"→ Freeze backbone: {train_overrides['freeze']} couches")

    print(f"\n⏳ Chargement du modèle {MODEL_WEIGHTS}...")
    model = YOLO(MODEL_WEIGHTS)

    print("\n" + "=" * 70)
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 70)

    try:
        _ = model.train(**train_overrides)

        print("\n" + "=" * 70)
        print("✓ ENTRAÎNEMENT TERMINÉ")
        print("=" * 70)

        best_model_path = os.path.join('runs/detect/shahed136_training/weights/best.pt')
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, OUTPUT_MODEL_NAME)
            print(f"✓ Copie du meilleur poids → {OUTPUT_MODEL_NAME}")
            print(f"  (Original: {best_model_path})")
            return best_model_path
        else:
            print(f"⚠ best.pt introuvable: {best_model_path}")
            return None

    except Exception as e:
        print(f"\n❌ Erreur entraînement: {e}")
        import traceback; traceback.print_exc()
        return None

# ======================
#   Validation
# ======================

def validate_model(model_path: str, data_yaml_path: str):
    if not model_path or not os.path.exists(model_path):
        print("\n⚠ Modèle introuvable, validation annulée")
        return None

    print("\n" + "=" * 70)
    print("VALIDATION DU MODÈLE")
    print("=" * 70)

    model = YOLO(model_path)
    device_str, _ = select_device()

    metrics = model.val(
        data=data_yaml_path,
        split='val',
        imgsz=IMG_SIZE if not FAST_MODE else 448,
        device=device_str,
        plots=True,
        save_json=True
    )

    print("\n" + "=" * 70)
    print("📊 RÉSULTATS DE VALIDATION")
    print("=" * 70)
    print(f"→ mAP50:      {getattr(metrics.box, 'map50', float('nan')):.3f}")
    print(f"→ mAP50-95:   {getattr(metrics.box, 'map', float('nan')):.3f}")
    print(f"→ Précision:  {getattr(metrics.box, 'mp', float('nan')):.3f}")
    print(f"→ Rappel:     {getattr(metrics.box, 'mr', float('nan')):.3f}")
    return metrics

# ======================
#   Test rapide
# ======================

def quick_test(model_path: str):
    if not model_path or not os.path.exists(model_path):
        print("\n⚠ Modèle introuvable, test annulé")
        return

    print("\n" + "=" * 70)
    print("TEST RAPIDE SUR QUELQUES IMAGES (valid)")
    print("=" * 70)

    model = YOLO(model_path)
    test_images_dir = os.path.join(DATASET_PATH, "valid/images")
    if not os.path.exists(test_images_dir):
        print(f"⚠ Dossier introuvable: {test_images_dir}")
        return

    imgs = [os.path.join(test_images_dir, f)
            for f in os.listdir(test_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))][:5]
    if not imgs:
        print("⚠ Aucune image de test trouvée")
        return

    device_str, _ = select_device()
    results = model.predict(
        imgs,
        conf=0.25,
        save=True,
        project='runs/detect',
        name='quick_test',
        device=device_str
    )

    for r in results:
        img_name = os.path.basename(r.path)
        n = 0 if r.boxes is None else len(r.boxes)
        print(f"  {img_name}: {n} détection(s)")
        if r.boxes is not None:
            for b in r.boxes:
                cls = int(b.cls[0]); conf = float(b.conf[0])
                cname = r.names.get(cls, f"class_{cls}")
                print(f"    → {cname}: {conf:.2f}")

    print("✓ Images annotées: runs/detect/quick_test/")

# ======================
#   MAIN
# ======================

def main():
    print("=" * 70)
    print("🎯 YOLO11 - DÉTECTEUR SHAHED-136")
    print("=" * 70)

    data_yaml_path = fix_data_yaml()
    if not data_yaml_path:
        return

    if not verify_dataset():
        print("❌ Structure du dataset incomplète")
        return

    if not AUTO_START:
        resp = input("\n▶ Lancer l'entraînement maintenant ? (o/n): ").strip().lower()
        if resp != "o":
            print("\n❌ Entraînement annulé.")
            return

    best = train_model(data_yaml_path)
    if best is None:
        print("\n❌ Entraînement échoué")
        return

    validate_model(best, data_yaml_path)
    quick_test(best)

    print("\n" + "=" * 70)
    print("✅ FIN")
    print("=" * 70)
    print(f"→ Modèle final: {OUTPUT_MODEL_NAME}")
    print("→ Logs & courbes: runs/detect/shahed136_training/")

if __name__ == "__main__":
    main()
