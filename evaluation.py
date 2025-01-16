import os
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import label, center_of_mass
import nibabel as nib


def dice_score(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

def iou_score(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

def sensitivity(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def ppv(y_pred, y_true):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

# Filtrar objetos pequeños
def filter_small_objects(labeled_img, min_size):
    labeled_img = labeled_img.astype(int)  # Asegurar tipo entero
    sizes = np.bincount(labeled_img.ravel())    
    mask = sizes >= min_size
    mask[0] = False  # Mantén el fondo en 0
    return mask[labeled_img]

def calculate_nodule_metrics_IoU(pred_img, gt_img, iou_threshold=0.1, min_size=10):

    pred_labeled, num_pred = label(pred_img)
    # gt_labeled, num_gt = label(gt_img)
    
    gt_labeled = gt_img
    num_gt = 1  # Considerando que solo debería haber un gt

    pred_labeled = filter_small_objects(pred_labeled, min_size)
    gt_labeled = filter_small_objects(gt_labeled, min_size)

    # Recalcular etiquetas después del filtrado
    pred_labeled, num_pred = label(pred_labeled)
    # gt_labeled, num_gt = label(gt_labeled)

    matched_pred = set()
    matched_gt = set()

    for gt_idx in range(1, num_gt + 1):
        gt_mask = gt_labeled == gt_idx

        best_iou = 0
        best_pred_idx = None

        for pred_idx in range(1, num_pred + 1):
            if pred_idx in matched_pred:
                continue

            pred_mask = pred_labeled == pred_idx

            intersection = np.sum(gt_mask & pred_mask)
            union = np.sum(gt_mask | pred_mask)

            if union > 0:
                iou = intersection / union
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_pred_idx = pred_idx

        if best_pred_idx is not None:
            matched_pred.add(best_pred_idx)
            matched_gt.add(gt_idx)

    false_positives = num_pred - len(matched_pred)
    false_negatives = num_gt - len(matched_gt)

    if false_negatives / num_gt == 0.5:
        print("mal van las label...")

    return {
        "false_positive_rate_iou": false_positives / num_pred if num_pred > 0 else 0,
        "false_negative_rate_iou": false_negatives / num_gt if num_gt > 0 else 0,
    }


def calculate_nodule_metrics_centr(pred_img, gt_img, distance_threshold=5, min_size=10):
    # Binarización explícita
    pred_img = (pred_img > 0).astype(int)
    gt_img = (gt_img > 0).astype(int)

    # Etiquetado de componentes conectados
    pred_labeled, num_pred = label(pred_img)
    # gt_labeled, num_gt = label(gt_img)
    gt_labeled = gt_img
    num_gt = 1


    pred_labeled = filter_small_objects(pred_labeled, min_size)
    gt_labeled = filter_small_objects(gt_labeled, min_size)

    # Recalcular etiquetas después del filtrado
    pred_labeled, num_pred = label(pred_labeled)
    # gt_labeled, num_gt = label(gt_labeled)

    matched_pred = set()
    matched_gt = set()

    pred_centroids = center_of_mass(pred_img, pred_labeled, range(1, num_pred + 1))
    gt_centroids = center_of_mass(gt_img, gt_labeled, range(1, num_gt + 1))
    
    if len(gt_centroids) > 1:
        print("")

    for gt_idx, gt_centroid in enumerate(gt_centroids, start=1):
        best_distance = float('inf')
        best_pred_idx = None

        for pred_idx, pred_centroid in enumerate(pred_centroids, start=1):
            if pred_idx in matched_pred:
                continue

            distance = np.linalg.norm(np.array(gt_centroid) - np.array(pred_centroid))

            if distance <= distance_threshold and distance < best_distance:
                best_distance = distance
                best_pred_idx = pred_idx

        if best_pred_idx is not None:
            matched_pred.add(best_pred_idx)
            matched_gt.add(gt_idx)

    false_positives = num_pred - len(matched_pred)
    false_negatives = num_gt - len(matched_gt)

    return {
        "false_positive_rate_centr": false_positives / num_pred if num_pred > 0 else 0,
        "false_negative_rate_centr": false_negatives / num_gt if num_gt > 0 else 0,
    }
    
def calculate_metrics(pred_dir, gt_dir):
    pred_files = os.listdir(pred_dir)
    gt_files = os.listdir(gt_dir)

    count = 0
    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    ppv_scores = []
    false_positive_rates_iou = []
    false_negative_rates_iou = []
    
    false_positive_rates_centr = []
    false_negative_rates_centr = []

    try:
        for pred_file in pred_files:
            pred_file_name = pred_file.replace("_pred", "")

            # Busca el archivo correspondiente en gt_files
            matching_gt_file = next((gt_file for gt_file in gt_files if gt_file == pred_file_name), None)

            if matching_gt_file:
                count += 1
                pred_path = os.path.join(pred_dir, pred_file)
                gt_path = os.path.join(gt_dir, matching_gt_file)

                if pred_path.endswith(".npy"):
                    pred_img = np.load(pred_path)
                    gt_img = np.load(gt_path)
                else:
                    pred_img = nib.load(pred_path).get_fdata()
                    gt_img = nib.load(gt_path).get_fdata()

                if np.sum(gt_img) == 0 and np.sum(pred_img) == 0:
                    continue

                dice = dice_score(pred_img, gt_img)
                iou = iou_score(pred_img, gt_img)
                sen = sensitivity(pred_img, gt_img)
                ppv_value = ppv(pred_img, gt_img)

                nodule_metrics_iou = calculate_nodule_metrics_IoU(pred_img, gt_img)
                nodule_metrics_centr = calculate_nodule_metrics_centr(pred_img, gt_img)

                dice_scores.append(dice)
                iou_scores.append(iou)
                sensitivity_scores.append(sen)
                ppv_scores.append(ppv_value)
                false_positive_rates_iou.append(nodule_metrics_iou["false_positive_rate_iou"])
                false_positive_rates_centr.append(nodule_metrics_centr["false_positive_rate_centr"])
                false_negative_rates_iou.append(nodule_metrics_iou["false_negative_rate_iou"])
                false_negative_rates_centr.append(nodule_metrics_centr["false_negative_rate_centr"])
            else:
                print(f"No match found for {pred_file}")
                
    except Exception as e:
        print(e)

    metrics = {
        "dice_scores": dice_scores,
        "iou_scores": iou_scores,
        "sensitivity_scores": sensitivity_scores,
        "ppv_scores": ppv_scores,
        "false_positive_rates_iou": false_positive_rates_iou,
        "false_negative_rates_iou": false_negative_rates_iou,
        "false_positive_rates_centr": false_positive_rates_centr,
        "false_negative_rates_centr": false_negative_rates_centr,
    }
    print("Count:", count)
    return metrics


def save_boxplot(metrics, experiment_name):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f'Diagramas de caja para {experiment_name}')

    metric_names = [
        "Dice Score",
        "IoU Score",
        "Sensitivity",
        "PPV",
        "False Positive Rate IOU",
        "False Negative Rate IOU",
    ]
    data = [
        metrics["dice_scores"],
        metrics["iou_scores"],
        metrics["sensitivity_scores"],
        metrics["ppv_scores"],
        metrics["false_positive_rates_iou"],
        metrics["false_negative_rates_iou"],
        metrics["false_positive_rates_centr"],
        metrics["false_negative_rates_centr"],
    ]
    
    ax_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for ax_pos, metric_name, metric_data in zip(ax_positions, metric_names, data):
        ax = axs[ax_pos[0], ax_pos[1]]
        ax.boxplot(metric_data)
        ax.set_title(metric_name)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    boxplot_path = f"./results/plots/boxplots_{experiment_name}.png"
    plt.savefig(boxplot_path)
    plt.close(fig)

    return boxplot_path

def main():
    pred_dir = "/app/data/LIDC-IDRI/nnUnet/predicts/"
    gt_dir = "/app/data/LIDC-IDRI/nnUnet/labels/"
    model_name = "weights/model_nnU-Net.pt"

    actual_date = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = f"evaluation_{actual_date}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        try:
            metrics = calculate_metrics(pred_dir, gt_dir)
            mean_std_metrics = {
                "Mean Dice": np.mean(metrics["dice_scores"]),
                "Std Dice": np.std(metrics["dice_scores"]),
                "Mean IoU": np.mean(metrics["iou_scores"]),
                "Std IoU": np.std(metrics["iou_scores"]),
                "Mean Sensitivity": np.mean(metrics["sensitivity_scores"]),
                "Std Sensitivity": np.std(metrics["sensitivity_scores"]),
                "Mean PPV": np.mean(metrics["ppv_scores"]),
                "Std PPV": np.std(metrics["ppv_scores"]),
                "FPR_IoU": np.mean(metrics["false_positive_rates_iou"]),
                "FNR_IoU": np.mean(metrics["false_negative_rates_iou"]),
                "FPR_centr": np.mean(metrics["false_positive_rates_centr"]),
                "FNR_centr": np.mean(metrics["false_negative_rates_centr"]),
            }
            for key, value in mean_std_metrics.items():
                print(f"{key}: {value:.4f}")
                mlflow.log_metric(key, value)
            mlflow.log_param("Prediction Directory", pred_dir)
            mlflow.log_param("Ground Truth Directory", gt_dir)
            mlflow.log_param("Model name", model_name)
            boxplot_path = save_boxplot(metrics, experiment_name)
            mlflow.log_artifact(boxplot_path)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            mlflow.log_param("Error", str(e))


if __name__ == "__main__":
    main()
