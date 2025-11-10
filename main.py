"""
Détection Shahed-136 avec mesure de distance + Tracking intelligent
Utilise le modèle YOLOv11 custom entraîné (mAP50: 99.5%)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import argparse
import os

# ======================
#   CONFIGURATION
# ======================

# Modèle entraîné (99.5% mAP50 !)
DEFAULT_MODEL = "shahed136_detector.pt"
FALLBACK_MODEL = "yolo11n.pt"

# Dimensions réelles Shahed-136 (vérifiées)
SHAHED_LENGTH_CM = 350.0   # 3.5 m
SHAHED_SPAN_CM = 250.0     # 2.5 m
SHAHED_HEIGHT_CM = 60.0    # ~60 cm

# Paramètres de détection ÉQUILIBRÉS + ANTI-ARTEFACTS
MIN_CONF = 0.35  # Confiance moyenne augmentée (55%)
MIN_AREA = 1  # Aire minimale (ANTI petits artefacts)
MAX_AREA = 4000  # Aire maximale (ANTI gros artefacts)

# Paramètres de TRACKING (suivi temporal)
STABILITY_FRAMES = 2  # Détection stable sur N frames pour valider
MEMORY_FRAMES = 5     # Garde la dernière position pendant N frames si perdue
IOU_THRESHOLD = 0.3   # Seuil IoU pour considérer "même objet"

# Paramètres de DISTANCE (calibration)
FOV_CAMERA = 60           # Field of View de la caméra (degrés) - AJUSTABLE
CALIBRATION_FACTOR = 1.0  # Facteur de correction manuel (1.0 = neutre)
USE_ADVANCED_PNP = False  # DÉSACTIVÉ temporairement (cause valeurs aberrantes)
DISTANCE_SMOOTHING = 20   # Lissage fort (20 frames)
MAX_REASONABLE_DISTANCE = 5000  # Distance max raisonnable en mètres
MIN_REASONABLE_DISTANCE = 0.5   # Distance min raisonnable en mètres
OUTLIER_REJECTION_FACTOR = 2.5  # Facteur pour rejeter outliers (écart > 2.5× médiane)

# Vidéo
VIDEO_SOURCE = "booo.mp4"  # Chemin vidéo par défaut
FRAME_W, FRAME_H = 1280, 720

# Classe détectée: 'military-drone' (ID: 0)
SHAHED_CLASS_ID = 0

# ======================
#   CHARGEMENT MODÈLE
# ======================

def load_model(model_path=None):
    """Charge le modèle YOLOv11 custom."""
    
    if model_path is None:
        model_path = DEFAULT_MODEL
    
    print("="*70)
    print("CHARGEMENT DU MODÈLE CUSTOM")
    print("="*70)
    
    if os.path.exists(model_path):
        print(f"✓ Modèle trouvé: {model_path}")
        model = YOLO(model_path)
        
        # Affichage des classes
        if hasattr(model, 'names'):
            print(f"\nClasses disponibles:")
            for idx, name in model.names.items():
                print(f"  [{idx}] {name}")
        
        print(f"\n✓ Modèle chargé avec succès")
        print(f"  → Classe détectée: '{model.names[SHAHED_CLASS_ID]}'")
        
        return model
    
    else:
        print(f"❌ Modèle introuvable: {model_path}")
        print(f"→ Utilisez: python3 train_shahed_model.py")
        return None


# ======================
#   TRACKER SIMPLE
# ======================

class SimpleTracker:
    """
    Tracker simple pour suivre un drone entre les frames.
    Évite les pertes temporaires et le compteur qui s'emballe.
    """
    def __init__(self, iou_threshold=0.3, memory_frames=5):
        self.last_bbox = None
        self.frames_since_detection = 0
        self.memory_frames = memory_frames
        self.iou_threshold = iou_threshold
        self.tracking_id = 0
        self.is_tracking = False
        
    def iou(self, box1, box2):
        """Calcule l'IoU entre deux bboxes [x1,y1,x2,y2]."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detection):
        """
        Met à jour le tracker avec une nouvelle détection.
        Retourne (is_new_object, tracking_status).
        """
        if detection is None:
            # Pas de détection
            self.frames_since_detection += 1
            
            if self.frames_since_detection > self.memory_frames:
                # Perte définitive
                self.last_bbox = None
                self.is_tracking = False
                return (False, "PERDU")
            else:
                # En mémoire temporaire
                return (False, "MÉMOIRE")
        
        else:
            bbox = detection["bbox"]
            
            if self.last_bbox is None:
                # Premier objet détecté
                self.last_bbox = bbox
                self.frames_since_detection = 0
                self.tracking_id += 1
                self.is_tracking = True
                return (True, "NOUVEAU")
            
            else:
                # Vérifier si c'est le même objet
                iou_score = self.iou(self.last_bbox, bbox)
                
                if iou_score >= self.iou_threshold:
                    # Même objet
                    self.last_bbox = bbox
                    self.frames_since_detection = 0
                    self.is_tracking = True
                    return (False, "SUIVI")
                
                else:
                    # Nouvel objet (ou saut trop grand)
                    self.last_bbox = bbox
                    self.frames_since_detection = 0
                    self.tracking_id += 1
                    self.is_tracking = True
                    return (True, "NOUVEAU")
    
    def get_predicted_bbox(self):
        """Retourne la dernière bbox connue (pour affichage en mémoire)."""
        return self.last_bbox


# ======================
#   MODÈLE 3D GÉOMÉTRIQUE
# ======================

def create_geometric_model():
    """Crée le modèle 3D géométrique du Shahed-136."""
    L = SHAHED_LENGTH_CM
    W = SHAHED_SPAN_CM
    H = SHAHED_HEIGHT_CM
    
    keypoints_3d = {
        'nose': np.array([L/2, 0, 0], dtype=np.float32),
        'tail': np.array([-L/2, 0, 0], dtype=np.float32),
        'center': np.array([0, 0, 0], dtype=np.float32),
        'wing_left_tip': np.array([0, -W/2, 0], dtype=np.float32),
        'wing_right_tip': np.array([0, W/2, 0], dtype=np.float32),
        'wing_left_root': np.array([L/6, -W/6, 0], dtype=np.float32),
        'wing_right_root': np.array([L/6, W/6, 0], dtype=np.float32),
        'top': np.array([0, 0, H/2], dtype=np.float32),
        'bottom': np.array([0, 0, -H/2], dtype=np.float32),
    }
    
    return keypoints_3d


# ======================
#   DÉTECTION
# ======================

def detect_target(model, frame, min_conf=0.55):
    """
    Détecte un Shahed-136/military-drone dans la frame.
    VERSION ÉQUILIBRÉE : compromis sensibilité/précision.
    """
    results = model.predict(
        source=frame, 
        stream=False, 
        verbose=False, 
        imgsz=640,
        conf=min_conf,
        iou=0.5,
        max_det=8
    )
    
    candidates = []
    
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            
            # Filtre classe
            if cls != SHAHED_CLASS_ID:
                continue
            
            # Filtres géométriques modérés
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            if area < MIN_AREA or area > MAX_AREA:
                continue
            
            if h <= 0:
                continue
            
            aspect = w / h
            # Aspect ratio modéré (exclut formes extrêmes)
            if not (0.8 < aspect < 4.5):
                continue
            
            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "cls": cls,
                "class_name": model.names.get(cls, f"class_{cls}"),
                "area": area,
                "aspect": aspect
            })
    
    if not candidates:
        return None
    
    # Meilleur = confiance max
    best = max(candidates, key=lambda x: x["conf"])
    
    return best


# ======================
#   CAMÉRA
# ======================

def estimate_camera_matrix(frame_w, frame_h, fov_deg=60):
    """Estime la matrice intrinsèque de la caméra."""
    focal_length = (frame_w / 2) / np.tan(np.radians(fov_deg / 2))
    camera_matrix = np.array([
        [focal_length, 0, frame_w / 2],
        [0, focal_length, frame_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


# ======================
#   EXTRACTION POINTS 2D
# ======================

def extract_keypoints_2d(frame, bbox):
    """Extrait les points caractéristiques 2D de l'objet détecté."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarisation adaptative
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphologie pour nettoyer
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Détection contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Plus grand contour
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 30:
        return None
    
    # Points extrêmes
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    
    # Centroïde
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx = (rightmost[0] + leftmost[0]) // 2
        cy = (topmost[1] + bottommost[1]) // 2
    
    keypoints_2d = {
        'center': np.array([cx + x1, cy + y1], dtype=np.float32),
        'left': np.array([leftmost[0] + x1, leftmost[1] + y1], dtype=np.float32),
        'right': np.array([rightmost[0] + x1, rightmost[1] + y1], dtype=np.float32),
        'top': np.array([topmost[0] + x1, topmost[1] + y1], dtype=np.float32),
        'bottom': np.array([bottommost[0] + x1, bottommost[1] + y1], dtype=np.float32)
    }
    
    return keypoints_2d


# ======================
#   DISTANCE (PnP 3D)
# ======================

def estimate_distance_pnp(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs, bbox):
    """
    Estime la distance via solvePnP (Perspective-n-Point).
    Méthode la plus précise pour mesure de distance 3D.
    """
    if keypoints_2d is None:
        return None
    
    x1, y1, x2, y2 = bbox
    bbox_aspect = (x2 - x1) / max(1, (y2 - y1))
    
    # Choix des correspondances 3D↔2D selon l'angle de vue
    if bbox_aspect > 2.5:  # Vue latérale (drone de profil)
        mappings = [
            ('center', 'center'),
            ('wing_left_tip', 'left'),
            ('wing_right_tip', 'right'),
            ('nose', 'top'),
            ('tail', 'bottom')
        ]
    elif bbox_aspect < 1.3:  # Vue frontale (face)
        mappings = [
            ('center', 'center'),
            ('wing_left_tip', 'left'),
            ('wing_right_tip', 'right'),
            ('top', 'top'),
            ('bottom', 'bottom')
        ]
    else:  # Vue oblique (3/4)
        mappings = [
            ('center', 'center'),
            ('wing_left_tip', 'left'),
            ('wing_right_tip', 'right'),
            ('nose', 'top'),
            ('tail', 'bottom')
        ]
    
    object_points = []
    image_points = []
    
    for key3d, key2d in mappings:
        if key3d in keypoints_3d and key2d in keypoints_2d:
            object_points.append(keypoints_3d[key3d])
            image_points.append(keypoints_2d[key2d])
    
    if len(object_points) < 4:
        return None
    
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    # solvePnP avec RANSAC (robuste aux outliers)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        reprojectionError=8.0,  # Réduit pour plus de précision
        confidence=0.99,
        iterationsCount=200,    # Plus d'itérations pour meilleure convergence
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success or inliers is None or len(inliers) < 3:
        return None
    
    # Distance = norme du vecteur translation (position caméra → drone)
    distance_cm = np.linalg.norm(tvec)
    
    # Application du facteur de calibration
    distance_cm *= CALIBRATION_FACTOR
    
    # Conversion rotation en angles d'Euler (optionnel)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    return {
        'distance_cm': distance_cm,
        'distance_m': distance_cm / 100.0,
        'tvec': tvec,
        'rvec': rvec,
        'rotation_matrix': rotation_matrix,
        'inliers': len(inliers),
        'method': 'PnP-RANSAC'
    }


def estimate_distance_simple(bbox, camera_matrix, frame_shape):
    """
    Méthode géométrique AMÉLIORÉE : choix automatique dimension selon angle.
    Utilise plusieurs heuristiques pour plus de robustesse.
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    focal_px = camera_matrix[0, 0]
    
    # Aspect ratio pour déterminer l'angle de vue
    aspect = bbox_width / max(1, bbox_height)
    
    # AMÉLIORATION : Calcul multi-dimension avec poids
    distances_weighted = []
    
    # Dimension 1 : Largeur (vue latérale privilégiée)
    if aspect > 1.5:  # Vue plutôt latérale
        dist_width = (SHAHED_SPAN_CM * focal_px) / bbox_width
        distances_weighted.append((dist_width, 0.7))  # Poids fort
    else:
        dist_width = (SHAHED_SPAN_CM * focal_px) / bbox_width
        distances_weighted.append((dist_width, 0.3))  # Poids faible
    
    # Dimension 2 : Hauteur (vue frontale)
    if aspect < 1.8:  # Vue plutôt frontale/oblique
        # Hauteur visible = envergure si vue frontale
        apparent_height_cm = SHAHED_SPAN_CM if aspect < 1.3 else SHAHED_HEIGHT_CM * 3
        dist_height = (apparent_height_cm * focal_px) / bbox_height
        distances_weighted.append((dist_height, 0.6 if aspect < 1.3 else 0.4))
    
    # Dimension 3 : Diagonale (toujours calculée)
    diag_px = np.sqrt(bbox_width**2 + bbox_height**2)
    diag_real_cm = np.sqrt(SHAHED_SPAN_CM**2 + SHAHED_LENGTH_CM**2)
    dist_diag = (diag_real_cm * focal_px) / diag_px
    distances_weighted.append((dist_diag, 0.5))
    
    # Calcul de la distance pondérée
    total_weight = sum(w for _, w in distances_weighted)
    distance_cm = sum(d * w for d, w in distances_weighted) / total_weight
    
    # Application du facteur de calibration
    distance_cm *= CALIBRATION_FACTOR
    
    # Détermination du type de vue pour affichage
    if aspect > 2.2:
        view_type = "lateral"
    elif aspect < 1.2:
        view_type = "frontal"
    else:
        view_type = "oblique"
    
    return {
        'distance_cm': distance_cm,
        'distance_m': distance_cm / 100.0,
        'method': f'Géo-{view_type}',
        'aspect': aspect,
        'bbox_size': (bbox_width, bbox_height),
        'confidence': min(total_weight / len(distances_weighted), 1.0)
    }


# ======================
#   MAIN
# ======================

def main():
    # Arguments CLI
    parser = argparse.ArgumentParser(
        description='Détection Shahed-136 avec mesure de distance + Tracking (YOLOv11 custom)'
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Chemin modèle .pt (défaut: shahed136_detector.pt)')
    parser.add_argument('--video', type=str, default=None,
                       help='Vidéo source (défaut: booo.mp4)')
    parser.add_argument('--conf', type=float, default=0.55,
                       help='Confiance minimale (défaut: 0.55 - ANTI-ARTEFACTS)')
    parser.add_argument('--fov', type=float, default=60,
                       help='FOV caméra en degrés (défaut: 60)')
    parser.add_argument('--calib', type=float, default=1.0,
                       help='Facteur calibration distance (défaut: 1.0)')
    parser.add_argument('--debug', action='store_true',
                       help='Mode debug (affiche calculs intermédiaires)')
    args = parser.parse_args()
    
    # Configuration
    min_conf = args.conf
    video_source = args.video if args.video else VIDEO_SOURCE
    global FOV_CAMERA, CALIBRATION_FACTOR
    FOV_CAMERA = args.fov
    CALIBRATION_FACTOR = args.calib
    debug_mode = args.debug
    
    print("="*70)
    print("DÉTECTION SHAHED-136 + MESURE DE DISTANCE 3D + TRACKING")
    print("Modèle: YOLOv11 custom (mAP50: 99.5%)")
    print("="*70)
    
    # Chargement modèle
    model = load_model(args.model)
    
    if model is None:
        print("\n❌ Impossible de charger le modèle")
        return
    
    # Modèle 3D géométrique
    keypoints_3d = create_geometric_model()
    print(f"\n✓ Modèle 3D créé:")
    print(f"  - Longueur: {SHAHED_LENGTH_CM/100:.2f}m")
    print(f"  - Envergure: {SHAHED_SPAN_CM/100:.2f}m")
    print(f"  - Hauteur: {SHAHED_HEIGHT_CM/100:.2f}m")
    
    # Ouverture vidéo
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)
    
    if isinstance(video_source, str):
        print(f"\n→ Vidéo: {video_source}")
    else:
        print(f"\n→ Webcam: index {video_source}")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir: {video_source}")
        return
    
    # Configuration résolution (webcam uniquement)
    if isinstance(video_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"→ Résolution: {actual_w}x{actual_h}")
    if total_frames > 0:
        print(f"→ Frames totales: {total_frames}")
        print(f"→ FPS vidéo: {fps_video:.1f}")
    
    # Matrice caméra
    camera_matrix, dist_coeffs = estimate_camera_matrix(actual_w, actual_h, FOV_CAMERA)
    print(f"→ Focale estimée: {camera_matrix[0,0]:.1f} px (FOV: {FOV_CAMERA}°)")
    print(f"→ Confiance minimale: {min_conf}")
    print(f"→ Facteur calibration: {CALIBRATION_FACTOR}")
    if debug_mode:
        print(f"→ MODE DEBUG ACTIVÉ")
    
    print("\n" + "="*70)
    print("[Q: Quitter | ESPACE: Pause | R: Reset stats]")
    print("="*70 + "\n")
    
    # Stats
    t_prev = time.time()
    fps_hist = deque(maxlen=15)
    frame_count = 0
    unique_drones_count = 0
    distance_history = deque(maxlen=DISTANCE_SMOOTHING)
    distance_raw_history = deque(maxlen=5)  # Historique brut pour détecter outliers
    
    # Tracker
    tracker = SimpleTracker(iou_threshold=IOU_THRESHOLD, memory_frames=MEMORY_FRAMES)
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("\n→ Fin de la vidéo")
            break
        
        frame_count += 1
        
        # FPS
        t_now = time.time()
        fps_hist.append(1.0 / max(1e-6, t_now - t_prev))
        t_prev = t_now
        fps = np.mean(fps_hist) if fps_hist else 0.0
        
        # Détection
        det = detect_target(model, frame, min_conf)
        
        # Update tracker
        is_new, tracking_status = tracker.update(det)
        
        if is_new:
            unique_drones_count += 1  # Nouveau drone unique
        
        if det:
            bbox = det["bbox"]
            conf = det["conf"]
            class_name = det["class_name"]
            
            # Extraction keypoints 2D
            keypoints_2d = extract_keypoints_2d(frame, bbox)
            
            # Estimation distance (GÉOMÉTRIQUE UNIQUEMENT pour stabilité)
            result = None
            if USE_ADVANCED_PNP:
                result = estimate_distance_pnp(
                    keypoints_3d, keypoints_2d,
                    camera_matrix, dist_coeffs, bbox
                )
                
                # Filtre de sécurité : rejette valeurs aberrantes du PnP
                if result and result['distance_m'] > MAX_REASONABLE_DISTANCE:
                    if debug_mode:
                        print(f"  ⚠ PnP aberrant ({result['distance_m']:.1f}m), fallback géométrique")
                    result = None
            
            # Méthode géométrique (toujours comme fallback ou unique)
            if result is None:
                result = estimate_distance_simple(bbox, camera_matrix, frame.shape)
            
            # Filtre final de sécurité
            if result and result['distance_m'] > MAX_REASONABLE_DISTANCE:
                if debug_mode:
                    print(f"  ⚠ Distance rejetée: {result['distance_m']:.1f}m (> {MAX_REASONABLE_DISTANCE}m)")
                result = None
            
            # Debug mode
            if debug_mode and result:
                print(f"\n[Frame {frame_count}] Distance BRUTE: {result['distance_m']:.2f}m")
                print(f"  Méthode: {result['method']}")
                if 'aspect' in result:
                    print(f"  Aspect ratio: {result['aspect']:.2f}")
                if 'bbox_size' in result:
                    w, h = result['bbox_size']
                    print(f"  Bbox: {w:.0f}x{h:.0f}px")
                if 'confidence' in result:
                    print(f"  Confiance: {result['confidence']:.2f}")
            
            # Affichage
            x1, y1, x2, y2 = [int(v) for v in bbox]
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            aspect = bbox_width / max(1, bbox_height)
            
            if result:
                dist_m = result['distance_m']
                method = result['method']
                
                # FILTRE ANTI-OUTLIERS AVANCÉ
                is_outlier = False
                
                # Vérification 1 : Bornes absolues
                if not (MIN_REASONABLE_DISTANCE <= dist_m <= MAX_REASONABLE_DISTANCE):
                    is_outlier = True
                    if debug_mode:
                        print(f"  ⚠ Outlier (hors bornes): {dist_m:.1f}m")
                
                # Vérification 2 : Écart par rapport à l'historique
                if len(distance_history) >= 3 and not is_outlier:
                    median_hist = np.median(distance_history)
                    deviation = abs(dist_m - median_hist) / (median_hist + 1e-6)
                    
                    if deviation > OUTLIER_REJECTION_FACTOR:
                        is_outlier = True
                        if debug_mode:
                            print(f"  ⚠ Outlier (écart {deviation:.1f}× médiane {median_hist:.1f}m)")
                
                # Ajout dans l'historique (uniquement si pas outlier)
                if not is_outlier:
                    distance_history.append(dist_m)
                    distance_raw_history.append(dist_m)
                elif debug_mode:
                    print(f"  → Distance rejetée, utilisation médiane historique")
                
                # Calcul de la distance lissée
                if len(distance_history) >= 3:
                    # Moyenne pondérée : 70% médiane + 30% valeur actuelle
                    dist_median = np.median(distance_history)
                    
                    if not is_outlier:
                        dist_smooth = 0.7 * dist_median + 0.3 * dist_m
                    else:
                        dist_smooth = dist_median  # Utilise uniquement historique si outlier
                    
                    # Calcul intervalle de confiance (5e-95e percentile)
                    dist_min = np.percentile(distance_history, 5)
                    dist_max = np.percentile(distance_history, 95)
                else:
                    dist_smooth = dist_m
                    dist_min = dist_m
                    dist_max = dist_m
                
                if debug_mode and len(distance_history) >= 3:
                    print(f"  → Distance LISSÉE: {dist_smooth:.2f}m (IC: {dist_min:.1f}-{dist_max:.1f}m)")
                
                # Couleur selon méthode
                if method.startswith('PnP'):
                    color = (0, 255, 0)  # Vert
                elif 'span' in method or 'lateral' in method.lower():
                    color = (0, 200, 255)  # Orange (vue latérale)
                elif 'front' in method:
                    color = (255, 150, 0)  # Bleu-orange (vue frontale)
                else:
                    color = (0, 165, 255)  # Orange standard
                
                # Rectangle détection
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Texte principal: distance avec intervalle de confiance
                if len(distance_history) >= 5:
                    text_main = f"{class_name}: {dist_smooth:.1f}m ({dist_min:.0f}-{dist_max:.0f}m)"
                else:
                    text_main = f"{class_name}: {dist_smooth:.1f}m"
                
                cv2.putText(frame, text_main, (x1, max(0, y1-15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Infos complémentaires
                info = f"Conf: {conf:.2f} | {method} | Status: {tracking_status}"
                if 'confidence' in result:
                    info += f" | Q: {result['confidence']:.2f}"
                if method.startswith('PnP') and 'inliers' in result:
                    info += f" ({result['inliers']} pts)"
                
                cv2.putText(frame, info, (x1, max(0, y1-40)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Visualisation points clés (désactivé si géométrique seul)
                if keypoints_2d and method.startswith('PnP'):
                    for key, pt in keypoints_2d.items():
                        cv2.circle(frame, tuple(pt.astype(int)), 4, (255, 0, 255), -1)
        
        # HUD (Heads-Up Display)
        status_color = (0, 255, 0) if tracker.is_tracking else (0, 0, 255)
        status_text = f"SUIVI ACTIF (ID:{tracker.tracking_id})" if tracker.is_tracking else "RECHERCHE..."
        
        cv2.putText(frame, status_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # Infos performance
        progress = f"{frame_count}"
        if total_frames > 0:
            progress += f"/{total_frames} ({100*frame_count/total_frames:.1f}%)"
        
        cv2.putText(frame, f"FPS: {fps:.1f} | Frame: {progress}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Compteur de drones UNIQUES
        cv2.putText(frame, f"Drones detectes: {unique_drones_count} | Tracking: {tracking_status}",
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Affichage
        cv2.imshow("Shahed-136 Detector [YOLOv11 Custom - 99.5% mAP50]", frame)
        
        # Contrôles clavier
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)  # Pause
        elif key == ord('r'):
            # Reset stats
            unique_drones_count = 0
            frame_count = 0
            distance_history.clear()
            distance_raw_history.clear()
            tracker = SimpleTracker(iou_threshold=IOU_THRESHOLD, memory_frames=MEMORY_FRAMES)
            print("\n→ Statistiques et tracker réinitialisés")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Rapport final
    print("\n" + "="*70)
    print("STATISTIQUES FINALES")
    print("="*70)
    print(f"→ Frames traitées: {frame_count}")
    print(f"→ Drones UNIQUES détectés: {unique_drones_count}")
    print(f"→ ID de tracking final: {tracker.tracking_id}")
    if fps_hist:
        print(f"→ FPS moyen: {np.mean(fps_hist):.1f}")
    if distance_history:
        print(f"→ Distance moyenne: {np.mean(distance_history):.1f}m")
        print(f"→ Distance médiane: {np.median(distance_history):.1f}m")
        print(f"→ Distance min/max: {min(distance_history):.1f}m / {max(distance_history):.1f}m")
        print(f"→ Écart-type: {np.std(distance_history):.1f}m")
        print(f"→ Coefficient variation: {100*np.std(distance_history)/np.mean(distance_history):.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()