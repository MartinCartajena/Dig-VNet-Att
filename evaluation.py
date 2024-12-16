import os
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime

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

def calculate_metrics(pred_dir, gt_dir):
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))
    
    dice_scores = []
    iou_scores = []
    sensitivity_scores = []
    ppv_scores = []
    
    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_file_name = pred_file.replace("_pred", "")
        if pred_file_name == gt_file:
            pred_path = os.path.join(pred_dir, pred_file)
            gt_path = os.path.join(gt_dir, gt_file)
            
            pred_img = np.load(pred_path)
            gt_img = np.load(gt_path)
            
            if np.sum(gt_img) == 0 and np.sum(pred_img) == 0:
                continue
            
            dice = dice_score(pred_img, gt_img)
            iou = iou_score(pred_img, gt_img)
            sen = sensitivity(pred_img, gt_img)
            ppv_value = ppv(pred_img, gt_img)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            sensitivity_scores.append(sen)
            ppv_scores.append(ppv_value)
    
    metrics = {
        "dice_scores": dice_scores,
        "iou_scores": iou_scores,
        "sensitivity_scores": sensitivity_scores,
        "ppv_scores": ppv_scores
    }
    return metrics

def save_boxplot(metrics, experiment_name):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Diagramas de caja para {experiment_name}')
    
    metric_names = ["Dice Score", "IoU Score", "Sensitivity", "PPV"]
    data = [metrics["dice_scores"], metrics["iou_scores"], metrics["sensitivity_scores"], metrics["ppv_scores"]]
    ax_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for ax_pos, metric_name, metric_data in zip(ax_positions, metric_names, data):
        ax = axs[ax_pos[0], ax_pos[1]]
        ax.boxplot(metric_data)
        ax.set_title(metric_name)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuste para título principal
    boxplot_path = f"./results/plots/boxplots_{experiment_name}.png"
    plt.savefig(boxplot_path)
    plt.close(fig)
    
    return boxplot_path

def main():
    # Directorios
    pred_dir = "./results/preds/LNDb_only_nod"
    gt_dir = "/app/data/labelsTs"
    model_name = "weights/model_aritz_data.pt"
    
    actual_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"evaluation_{actual_date}"
    mlflow.set_experiment(experiment_name)

    # Ejecuta el cálculo y guarda en MLflow
    with mlflow.start_run():
        try:
            metrics = calculate_metrics(pred_dir, gt_dir)
            
            # Calcula medias y desviaciones estándar
            mean_std_metrics = {
                "Mean Dice": np.mean(metrics["dice_scores"]),
                "Std Dice": np.std(metrics["dice_scores"]),
                "Mean IoU": np.mean(metrics["iou_scores"]),
                "Std IoU": np.std(metrics["iou_scores"]),
                "Mean Sensitivity": np.mean(metrics["sensitivity_scores"]),
                "Std Sensitivity": np.std(metrics["sensitivity_scores"]),
                "Mean PPV": np.mean(metrics["ppv_scores"]),
                "Std PPV": np.std(metrics["ppv_scores"])
            }
            
            # Imprime y guarda las métricas en MLflow
            for key, value in mean_std_metrics.items():
                print(f"{key}: {value:.4f}")
                mlflow.log_metric(key, value)
            
            # Guarda parámetros adicionales
            mlflow.log_param("Prediction Directory", pred_dir)
            mlflow.log_param("Ground Truth Directory", gt_dir)
            mlflow.log_param("Model name", model_name)
            
            # Genera y guarda el diagrama de caja en MLflow
            boxplot_path = save_boxplot(metrics, experiment_name)
            mlflow.log_artifact(boxplot_path)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            mlflow.log_param("Error", str(e))
            
main()
