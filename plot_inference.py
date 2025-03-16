import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns


def get_fps_res(inference_output_dir, training_mode):
    predicted_res_path = f"{inference_output_dir}/predicted_res_{training_mode}.py"
    target_res_path = f"{inference_output_dir}/target_res_{training_mode}.py"
    predicted_fps_path = f"{inference_output_dir}/predicted_fps_{training_mode}.py"
    target_fps_path = f"{inference_output_dir}/target_fps_{training_mode}.py"
    
    data = {}

    for file_path, key in zip(
        [predicted_res_path, target_res_path, predicted_fps_path, target_fps_path],
        ["predicted_res", "target_res", "predicted_fps", "target_fps"]
    ):
        with open(file_path, "r") as f:
            exec(f.read(), data)

    pred_res = np.array(data["predicted_res"])
    target_res = np.array(data["target_res"])
    pred_fps = np.array(data["predicted_fps"])
    target_fps = np.array(data["target_fps"])
    
    # count_target_1080 = np.sum(target_res == 1080)
    # count_predicted_1080 = np.sum(pred_res == 1080)
    # print(f'count_target_1080 {count_target_1080}, count_predicted_1080 {count_predicted_1080}')  # Output: 3
    return pred_res, target_res, pred_fps, target_fps

def quiver_plot_fps_res(training_mode, pred_res, target_res, pred_fps, target_fps, path='', SAVE=False, SHOW=False):
    # Compute quiver arrow components
    dx = target_res - pred_res
    dy = target_fps - pred_fps

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot quiver arrows where predicted != target
    mask = (dx != 0) | (dy != 0)
    ax.quiver(pred_res[mask], pred_fps[mask], dx[mask], dy[mask], angles='xy', scale_units='xy', scale=1, color='b', alpha=0.1,
              headwidth=8, headlength=10, headaxislength=8  # Make arrowhead bigger
              )

    # Plot points where predicted == target
    mask_identical = (dx == 0) & (dy == 0)
    ax.scatter(pred_res[mask_identical], pred_fps[mask_identical], color='r', label='Targets, predictions align')

    # Labels and formatting
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Framerate (FPS)")
    ax.set_title(f"Predicted vs Target FPS and Resolution (point to target) \n{datatype} {training_mode}")
    ax.grid(True)
    ax.legend()
    
    ax.set_xticks(resolution_ticks)
    ax.set_yticks(fps_ticks)

    if SAVE:
        plt.savefig(f'{path}_quiver.png')
    # if SHOW:
    #     print(f'show {SHOW}')
    # plt.show()

def scatter_plot_fps_res(training_mode, pred_res, target_res, pred_fps, target_fps, path='', SAVE=False, SHOW=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    jitter_strength = 2  # Adjust as needed

    target_fps_jittered = target_fps + np.random.uniform(-jitter_strength, jitter_strength, size=target_fps.shape)
    pred_fps_jittered = pred_fps + np.random.uniform(-jitter_strength, jitter_strength, size=pred_fps.shape)

    axes[0].scatter(target_fps_jittered, pred_fps_jittered, color='b', alpha=0.1, label='Predictions')
    axes[0].plot([min(target_fps), max(target_fps)], [min(target_fps), max(target_fps)], 'r--', label='Perfect Prediction (y=x)')
    axes[0].set_xlabel("Ground Truth FPS")
    axes[0].set_ylabel("Predicted FPS")
    axes[0].set_title("FPS Predicted vs. FPS GT")
    axes[0].set_xticks(fps_ticks)
    axes[0].set_yticks(fps_ticks)

    axes[0].legend()
    axes[0].grid(True)

    # --- 2. Scatter Plot: Resolution Predicted vs. Resolution GT ---
    jitter_strength = 30
    target_res_jittered = target_res + np.random.uniform(-jitter_strength, jitter_strength, size=target_res.shape)
    pred_res_jittered = pred_res + np.random.uniform(-jitter_strength, jitter_strength, size=pred_res.shape)

    axes[1].scatter(target_res_jittered, pred_res_jittered, color='g', alpha=0.1, label='Predictions')
    axes[1].plot([min(target_res), max(target_res)], [min(target_res), max(target_res)], 'r--', label='Perfect Prediction (y=x)')
    axes[1].set_xlabel("Ground Truth Resolution")
    axes[1].set_ylabel("Predicted Resolution")
    axes[1].set_title(f"Resolution Predicted vs. Resolution GT \n{training_mode}")
    axes[1].set_xticks(resolution_ticks)
    axes[1].set_yticks(resolution_ticks)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if SAVE:
        plt.savefig(f'{path}_scatter_plot.png')
    if SHOW:
        plt.show()



def heatmap_fps_res(pred_res, target_res, pred_fps, target_fps, path='', SAVE=False, SHOW=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fps_confusion_matrix = np.zeros((len(fps_ticks), len(fps_ticks)), dtype=int)

    # --- 2. FPS Heatmap ---
    for i, true_fps in enumerate(fps_ticks):
        for j, pred_fps_value in enumerate(fps_ticks):
            fps_confusion_matrix[i, j] = np.sum((target_fps == true_fps) & (pred_fps == pred_fps_value))

    fps_confusion_matrix = np.flipud(fps_confusion_matrix.T)

    sns.heatmap(fps_confusion_matrix, annot=True, cmap="Greens", fmt="d", 
                xticklabels=fps_ticks, yticklabels=fps_ticks[::-1], ax=axes[0])

    axes[0].set_xlabel("Ground Truth Resolution")
    axes[0].set_ylabel("Predicted Resolution")
    axes[0].set_title(f"Resolution Prediction Confusion Heatmap\n{datatype}")

    # --- 2. Resolution Heatmap ---
    resolution_confusion_matrix = np.zeros((len(resolution_ticks), len(resolution_ticks)), dtype=int)
    for i, true_res in enumerate(resolution_ticks):
        for j, pred_res_value in enumerate(resolution_ticks):
            resolution_confusion_matrix[i, j] = np.sum((target_res == true_res) & (pred_res == pred_res_value))

    resolution_confusion_matrix = np.flipud(resolution_confusion_matrix.T)

    sns.heatmap(resolution_confusion_matrix, annot=True, cmap="Greens", fmt="d", 
                xticklabels=resolution_ticks, yticklabels=resolution_ticks[::-1], ax=axes[1])

    axes[1].set_xlabel("Ground Truth Resolution")
    axes[1].set_ylabel("Predicted Resolution")
    axes[1].set_title(f"Resolution Prediction Confusion Heatmap\n{datatype}")

    if SAVE:
        plt.savefig(f'{path}_heatmap.png')
    # if SHOW:
    #     plt.show()


def scatter_plot_ground_truth(training_mode, pred_res, target_res, pred_fps, target_fps, path='', SAVE=False):
    jitter_strength = 2  # Adjust for more/less spread
    target_fps_jittered = target_fps + np.random.uniform(-jitter_strength, jitter_strength, size=target_fps.shape)
    jitter_strength = 5  # Adjust for more/less spread
    target_res_jittered = target_res + np.random.uniform(-jitter_strength * 5, jitter_strength * 5, size=target_res.shape)

    # Create a DataFrame with target FPS and resolution
    df = pd.DataFrame({"Resolution": target_res, "FPS": target_fps})
    # Count occurrences of each (Resolution, FPS) combination
    fps_res_counts = df.value_counts().reset_index()
    fps_res_counts.columns = ["Resolution", "FPS", "Count"]
    print(f'fps_res_counts \n{fps_res_counts}')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(target_res_jittered, target_fps_jittered, color='b', alpha=0.3, label='Target FPS')

    ax.set_xlabel("Resolution")
    ax.set_ylabel("FPS")
    ax.set_title("FPS vs Resolution Distribution")
    ax.grid(True)
    ax.legend()
    ax.set_xticks(resolution_ticks)
    ax.set_yticks(fps_ticks)

    # Show the plots
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    SAVE = True # True False
    SHOW = False 
    # 2025-03-07-frame-maxjod-sigmoid-excludeoutliers  2025-03-05-frame-dropjod-sigmoid-excludeoutliers
    # 2025-03-14-frame-dropjod-balanced-data 2025-03-14-frame-maxjod-balanced-data
    datatype = 'dropjod-balanced-data' 
    model_pth_parent_folder = Path(f'inference_outputs/2025-03-14-frame-{datatype}')
    # model_pth_parent_folder = Path('2025-03-07-frame-maxjod-sigmoid-excludeoutliers')
    fps_ticks = list(range(30, 121, 10))
    resolution_ticks = [360, 480, 720, 864, 1080]

    # Loop through each subfolder in the parent directory
    for subfolder in model_pth_parent_folder.iterdir():
        if subfolder.is_dir():  # like inference_outputs\2025-02-27-frame-dropjod-sigmoid\no_fps_no_resolution_no_velocity_09_06
            print(f'{subfolder.name} {subfolder.name}')  # like no_fps_no_resolution_13_04
            # if subfolder.name == "no_fps_no_resolution_03_20":
            training_mode = "_".join(subfolder.name.split("_")[:-2])
            pred_res, target_res, pred_fps, target_fps = get_fps_res(subfolder, training_mode)
            path = f'{subfolder}/{training_mode}'
            print(f'\npath {path}')
            # scatter_plot_ground_truth(training_mode, pred_res, target_res, pred_fps, target_fps)
            quiver_plot_fps_res(training_mode, pred_res, target_res, pred_fps, target_fps, path=path, SAVE=SAVE, SHOW=False)
            # scatter_plot_fps_res(training_mode, pred_res, target_res, pred_fps, target_fps, path=path, SAVE=SAVE, SHOW=False)

            # Count how many times predicted resolution and FPS matches target resolution and FPS
            res_match_counts = {res: np.sum((target_res == res) & (pred_res == res)) for res in resolution_ticks}
            fps_match_counts = {fps: np.sum((target_fps == fps) & (pred_fps == fps)) for fps in fps_ticks}

            res_match_df = pd.DataFrame(list(res_match_counts.items()), columns=['Resolution', 'Matches'])
            fps_match_df = pd.DataFrame(list(fps_match_counts.items()), columns=['FPS', 'Matches'])
            # print(f'res_match_df \n{res_match_df}')
            # print(f'fps_match_df \n{fps_match_df}')

            heatmap_fps_res(pred_res, target_res, pred_fps, target_fps, path=path, SAVE=SAVE, SHOW=True)

            #     break
            # else:
            #     continue


    # model_parent_folder = 'no_fps_no_resolution_13_04' # no_fps_no_resolution_no_velocity_09_06 no_fps_no_resolution_13_04
    # model_pth_parent_folder = f'2025-02-27-frame-dropjod-sigmoid'

    # training_mode = "no_fps_no_resolution"
    # input_folder = f'inference_outputs/{model_pth_parent_folder}/{model_parent_folder}'

    # SAVE = True # True False
    # path = f'{input_folder}/{training_mode}'
    # pred_res, target_res, pred_fps, target_fps = get_fps_res(input_folder, training_mode)
    # quiver_plot_fps_res(training_mode, pred_res, target_res, pred_fps, target_fps, path=path, SAVE=SAVE)
    # scatter_plot_fps_res(training_mode, pred_res, target_res, pred_fps, target_fps, path=path, SAVE=SAVE)
