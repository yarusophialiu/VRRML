# training all ablations
# python .\VideoQualityClassifier_velocity_local.py --training_mode no_fps
import subprocess

training_modes = [
                'no_fps_no_resolution_no_velocity',
                # 'no_fps', 
                # 'no_velocity', 
                # 'full',
                # 'no_res', 'no_fps_no_resolution', 
                #   'consecutive_patch', 'consecutive_patch_no_velocity', 
                # 'random_patch',
                #   'invariant_consecutive', 
                #   'invariant_random',
                  # 'invariant_consecutive_no_velocity', 
                #   'invariant_random_no_velocity'
                  ]
# drop jod
# training_modes = ['no_fps', 'consecutive_patch']

for mode in training_modes:
    print(f"===================== Starting training mode: {mode} =====================")
    subprocess.run(["python", "VideoQualityClassifier_velocity_local.py", "--training_mode", mode])
    print(f"Finished training mode: {mode}\n")
