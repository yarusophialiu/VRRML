import torch 

def set_manual_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) # Ensure CUDA also uses the same seed
    torch.cuda.manual_seed_all(42)  # If using multi-GPU
    np.random.seed(42) # Ensure reproducibility with NumPy and Python's random
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.benchmark = False


from pathlib import Path
from VideoQualityClassifier_velocity_test import evaluate_test_data
# from VideoQualityClassifier_velocity_local import set_manual_seed
from utils import *
from DeviceDataLoader import DeviceDataLoader
from VideoSinglePatchDataset import VideoSinglePatchDataset
from VideoDualPatchDataset import VideoDualPatchDataset
from VideoMultiplePatchDataset import VideoMultiplePatchDataset
from DecRefClassification_smaller import *
from DecRefClassification_dual_smaller import *
from DecRefClassification_multiple_smaller import *


def get_test_dataloader(ML_DATA_TYPE, batch_size, patch_size, device, parent_dir, FRAMENUMBER=True):
    PATCH_SIZE = 64
    if 'invariant_consecutive' in parent_dir:
        print(f'parent_dir {parent_dir} VideoMultiplePatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_consecutive_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoMultiplePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER) 
    elif 'invariant_random' in parent_dir:
        print(f'parent_dir {parent_dir} VideoMultiplePatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_random_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoMultiplePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    elif 'random' in parent_dir:
        print(f'parent_dir {parent_dir} VideoDualPatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_random_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoDualPatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    elif 'consecutive' in parent_dir:
        print(f'parent_dir {parent_dir} VideoDualPatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_consecutive_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoDualPatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    else:
        print(f'parent_dir {parent_dir} VideoSinglePatchDataset')
        data_test_folder = f'{ML_DATA_TYPE}/{velocity_type}/test_single_{PATCH_SIZE}x{PATCH_SIZE}' 
        test_dataset = VideoSinglePatchDataset(directory=f'{VRRML}/{data_test_folder}', min_bitrate=500, \
                                                max_bitrate=2000, patch_size=patch_size, VELOCITY=True, \
                                                VALIDATION=True, FRAMENUMBER=FRAMENUMBER)
    print(f'\nTest data size {len(test_dataset)}, patch_size {patch_size}, batch_size {batch_size}')
    print(f'data_test_folder {data_test_folder}\n')
    sample = test_dataset[0]
    print('sample image has ', sample['fps'], 'fps,', sample['resolution'], ' resolution,', sample['bitrate'], 'bps')
    print(f'sample velocity is {sample["velocity"]}') 
    print(f'sample path is {sample["path"]}') 
    test_dl = DataLoader(test_dataset, batch_size*2, shuffle = False, num_workers = 4, pin_memory = True)
    test_dl = DeviceDataLoader(test_dl, device) 
    return test_dl


def get_model(parent_dir, model_pth_path):
    model = None
    FPS, RESOLUTION, MODEL_VELOCITY = None, None, None
    if parent_dir.startswith('no_fps_no_resolution_no_velocity'): # with resolution
        FPS, RESOLUTION, MODEL_VELOCITY = False, False, False
        print(f'2 parent_dir {parent_dir}, DecRefClassification')
        model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    elif 'no_fps_no_resolution' in parent_dir:
        print(f'1 patch_type {parent_dir},  DecRefClassification')
        FPS, RESOLUTION, MODEL_VELOCITY = False, False, True
        model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    elif parent_dir.startswith('no_fps'): # with resolution
        FPS, RESOLUTION, MODEL_VELOCITY = False, True, True
        print(f'2 parent_dir {parent_dir}, DecRefClassification')
        model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    elif parent_dir.startswith('no_res'):
        FPS, RESOLUTION, MODEL_VELOCITY = True, False, True
        print(f'3 parent_dir {parent_dir}, DecRefClassification')
        model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    elif parent_dir.startswith('no_velocity'):
        # FPS, RESOLUTION, MODEL_VELOCITY = True, True, False
        FPS, RESOLUTION, MODEL_VELOCITY = False, False, False
        print(f'4 parent_dir {parent_dir}, DecRefClassification')
        model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    elif parent_dir.startswith('full'):
        FPS, RESOLUTION, MODEL_VELOCITY = True, True, True
        print(f'5 parent_dir {parent_dir}, DecRefClassification')
        model = DecRefClassification(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    

    elif parent_dir.startswith('consecutive_patch_no_velocity') or parent_dir.startswith('random_patch_no_velocity'):
        FPS, RESOLUTION, MODEL_VELOCITY = False, False, False
        print(f'6 parent_dir {parent_dir}, DecRefClassification_dual')
        model = DecRefClassification_dual(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    elif parent_dir.startswith('consecutive_patch') or parent_dir.startswith('random_patch'):
        FPS, RESOLUTION, MODEL_VELOCITY = False, False, True
        print(f'7 parent_dir{parent_dir}, DecRefClassification_dual')
        model = DecRefClassification_dual(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    # elif parent_dir.startswith('random_patch_no_velocity'):
    #     FPS, RESOLUTION, MODEL_VELOCITY = False, False, False
    #     print(f'6 parent_dir {parent_dir}, DecRefClassification_dual')
    #     model = DecRefClassification_dual(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    # elif parent_dir.startswith('random_patch') or parent_dir.startswith('random'):
    #     FPS, RESOLUTION, MODEL_VELOCITY = False, False, True
    #     print(f'7 parent_dir{parent_dir}, DecRefClassification_dual')
    #     model = DecRefClassification_dual(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    

    elif parent_dir.startswith('invariant_consecutive_no_velocity') or parent_dir.startswith('invariant_random_no_velocity'):
        FPS, RESOLUTION, MODEL_VELOCITY = False, False, False
        print(f'8 parent_dir {parent_dir}, DecRefClassification_multiple')
        model = DecRefClassification_multiple(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    elif parent_dir.startswith('invariant_consecutive') or parent_dir.startswith('invariant_random'):
        FPS, RESOLUTION, MODEL_VELOCITY = False, False, True
        print(f'9 parent_dir {parent_dir}, DecRefClassification_multiple')
        model = DecRefClassification_multiple(FPS=FPS, RESOLUTION=RESOLUTION, VELOCITY=MODEL_VELOCITY)
    print(f'FPS, RESOLUTION, MODEL_VELOCITY {FPS, RESOLUTION, MODEL_VELOCITY}')
    model.load_state_dict(torch.load(model_pth_path))
    return model, FPS, RESOLUTION, MODEL_VELOCITY


def get_training_mode(subfolder):
    # folder = parent_dir.split("\\")[1]
    arr = subfolder.split("_")[:-2]
    training_mode = "_".join(arr)
    print(f'training_mode {training_mode}')
    return training_mode


# loop through subfolders, test model, count model params, write to model.txt
if __name__ == "__main__":
    # Define the root folder where subfolders exist
    model_pth_parent_folder = f'2025-03-14-frame-maxjod' # TODO
    velocity_type = 'frame-velocity/max-jod-nooutliers' # TODO
    root_folder = Path(model_pth_parent_folder)
    ML_DATA_TYPE = 'ML'
    PATCH_SIZE = 64
    batch_size = 128 
    UPDATE_NUM_PARAMS = True
    UPDATE_TEST_RESULT = True
    SAVE_INFERENCE = True
    TEST_PERFORMANCE = True # False True
    # model_pth_path = f'{model_pth_parent_folder}/{model_parent_folder}/classification.pth' 

    device = get_default_device()
    set_manual_seed()

    for model_txt in root_folder.rglob("model.txt"): # <generator object Path.rglob at 0x000001D2BEA6C970>
        print(f'\n============= model_txt {model_txt} =============') # model_txt 2025-02-17\consecutive_patch_11_04\model.txt
        parent_dir = model_txt.parent
        # print(f'parent_dir {parent_dir.name}')
        model_pth = parent_dir / "classification.pth"
        print(f'model_pth {model_pth}')
        subfolder = str(parent_dir).split("\\")[1]

        test_dl = get_test_dataloader(ML_DATA_TYPE, batch_size * 10, (PATCH_SIZE, PATCH_SIZE), device, subfolder, FRAMENUMBER=True)
        model, FPS, RESOLUTION, MODEL_VELOCITY = get_model(subfolder, model_pth) 

        if UPDATE_NUM_PARAMS:
            # count parameters and update model.txt
            num_parameters = count_parameters(model_pth, model)
            if (model_txt).exists():  # Ensure classification.pth is in the same folder
                with model_txt.open("a") as file:
                    file.write(f"\nTotal number of parameters in the model: {num_parameters}\n")
                print(f"Updated: {model_txt}")

        if TEST_PERFORMANCE:
            model.to(device)  
            training_mode = get_training_mode(subfolder)
            result, res_preds, fps_preds, res_targets, fps_targets, jod_preds, jod_targets = evaluate_test_data(model, test_dl, training_mode)
            predicted_fps = torch.tensor([reverse_fps_map[int(pred)] for pred in fps_preds])
            target_fps = torch.tensor([reverse_fps_map[int(target)] for target in fps_targets])

            predicted_res = torch.tensor([reverse_res_map[int(pred)] for pred in res_preds])
            target_res = torch.tensor([reverse_res_map[int(target)] for target in res_targets])
            print(f'predicted_res {predicted_res}')
            print(f'target_res {target_res}')

            # Root Mean Square Error https://help.pecan.ai/en/articles/6456388-model-performance-metrics-for-regression-models
            resolution_RMSE = compute_RMSE(predicted_res, target_res)
            fps_RMSE = compute_RMSE(predicted_fps, target_fps)
            jod_RMSE = compute_RMSE(jod_preds, jod_targets)

            # Root Mean Squared Percentage Error (RMSPE)
            resolution_RMSEP = relative_error_metric(predicted_res, target_res) 
            fps_RMSEP = relative_error_metric(predicted_fps, target_fps) 
            print(f"FPS: Root Mean Squared Error {fps_RMSE}, Root Mean Squared Percentage Error (RMSPE): {fps_RMSEP}%")
            print(f"Resolution: Root Mean Squared Error {resolution_RMSE}, Root Mean Squared Percentage Error (RMSPE): {resolution_RMSEP}%\n")
            print(f'jod rmse {jod_RMSE}')
            print(f'test result \n {result}\n')


            if SAVE_INFERENCE:
                inference_output_dir = f'inference_outputs/{model_pth_parent_folder}/{parent_dir.name}'
                os.makedirs(inference_output_dir, exist_ok=True)
                with open(f"{inference_output_dir}/predicted_res_{training_mode}.py", "w") as f:
                    f.write(f"predicted_res = {predicted_res.tolist()}\n")
                with open(f"{inference_output_dir}/target_res_{training_mode}.py", "w") as f:
                    f.write(f"target_res = {target_res.tolist()}\n")
                with open(f"{inference_output_dir}/predicted_fps_{training_mode}.py", "w") as f:
                    f.write(f"predicted_fps = {predicted_fps.tolist()}\n")
                with open(f"{inference_output_dir}/target_fps_{training_mode}.py", "w") as f:
                    f.write(f"target_fps = {target_fps.tolist()}\n")


            if UPDATE_TEST_RESULT:
                if (model_txt).exists():  # Ensure classification.pth is in the same folder
                    with model_txt.open("a") as file:
                        file.write(f"FPS: Root Mean Squared Error {fps_RMSE}, Root Mean Squared Percentage Error (RMSPE): {fps_RMSEP}%\n")
                        file.write(f'Resolution: Root Mean Squared Error {resolution_RMSE}, Root Mean Squared Percentage Error (RMSPE): {resolution_RMSEP}%\n')
                        file.write(f'test result \n {result}\n\n')
                        file.write(f"{round(fps_RMSE, 1)} {round(fps_RMSEP)}%\n")
                        file.write(f'{round(resolution_RMSE, 1)} {round(resolution_RMSEP)}%\n')
                    print(f"Updated {model_txt}")
            
            # break
