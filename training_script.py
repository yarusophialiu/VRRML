# training all ablations
# python .\VideoQualityClassifier_velocity_local.py --training_mode no_fps
import subprocess

training_modes = ['no_fps', 'no_res', 'no_fps_no_resolution', 'no_velocity', 
                  'consecutive_patch', 'consecutive_patch_no_velocity', 'random_patch']

training_modes = ['no_fps', 'consecutive_patch']

for mode in training_modes:
    print(f"===================== Starting training mode: {mode} =====================")
    subprocess.run(["python", "VideoQualityClassifier_velocity_local.py", "--training_mode", mode])
    print(f"Finished training mode: {mode}\n")
