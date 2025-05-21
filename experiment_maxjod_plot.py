import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import binom
from datetime import datetime
  

def read_csv_value(folder_path):
    result_list = []
    for file_name in os.listdir(folder_path):
        # print(f'file_name {file_name}')
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path)
            result_list.extend(data.iloc[:, [2, 3, 5]].values.tolist())
    # print(result_list)
    return result_list


def process_csv_value(folder_path):
    x_positions = read_csv_value(folder_path)
    # print(f'x_positions {x_positions}')#  x_positions [[8000.0, 0.5, 1.0], [1000.0, 0.5, 1.0], [1000.0, 0.5, 1.0]
    nested_dict = {}
    for bitrate, speed, score in x_positions:
        if bitrate not in nested_dict:
            nested_dict[bitrate] = {}  # Initialize sub-dictionary for this bitrate
        if speed not in nested_dict[bitrate]:
            nested_dict[bitrate][speed] = []  # Initialize an empty list for this speed
        nested_dict[bitrate][speed].append(score)  # Append the score to the list for this speed

        # print(f'nested_dict {nested_dict}') # {2000.0: {0.5: [1.0, 0.0, 1.0, 1.0, ...], 1:[1,0...], 4000: {...}}
    return nested_dict


def get_y_range(nested_dict):
    y_range = {}
    expected_prob = {}
    for bitrate in bitrates:
        for speed in speeds:
            # initialize dictionary k v pair
            y_range[speed] = [] if speed not in y_range else y_range[speed]
            expected_prob[speed] = [] if speed not in expected_prob else expected_prob[speed]
            print(f'===== bitrate {bitrate}, speed {speed} =====')
            # compute binomial prob
            if bitrate in nested_dict and speed in nested_dict[bitrate]:
                p = sum(nested_dict[bitrate][speed]) / len(nested_dict[bitrate][speed]) # selected percentage
                # print(p)
                # expected_prob.append(p)
                k = binom.ppf(cumulative_threshold, N, p)
                # print(k/N)
                print(f'p {p}, k {k/N}')
                y_range[speed].append(tuple(k/N))
                expected_prob[speed].append(p)
    return y_range, expected_prob


if __name__ == "__main__":
    # Data definition
    # bitrates = ['1', '2', '4', '8']
    bitrates = [1000, 2000, 4000, 8000]
    speeds = [0.5, 1, 2] # ['v1', 'v2', 'v3']
    speeds_dict = {0.5: 'Slow', 1: 'Medium', 2: 'Fast'}
    bitrates_dict = {1000: 1, 2000: 2, 4000: 4, 8000: 8}
    bitrates_array = ['1Mbps ', '2Mbps ', '4Mbps ', '8Mbps ']
    separator_positions = [3, 6, 9, 12]  # Positions where separators are drawn

    # x_labels = [f"{speeds_dict[speed]}_{bitrates_dict[bitrate]}Mbps" for bitrate in bitrates for speed in speeds]
    x_labels = [f"{speeds_dict[speed]}" for bitrate in bitrates for speed in speeds]
    cumulative_threshold = [0.025, 0.975] # Cumulative probability threshold, 95% confidence interval
    N = 3 * 3 * 4 # Number of trials: 3 test scenes, 3 speeds, 4 bitrates, each scene has 3 paths?
    folder_path = r'C:\Users\15142\new\Falcor\Source\Samples\EncodeDecode\experimentResult-streaming-adaptive-approaches-official'

    colors = {
        0.5: "gold",
        1: "deepskyblue",
        2: "salmon"
    }

    # Prepare the x positions for the plot
    x_positions = list(range(len(x_labels)))
    print(f'x_labels {x_labels}') # x_labels ['Slow', 'Medium', 'Fast',
    # x_positions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


    PROCESS_CSV = True# True False
    if PROCESS_CSV:
        # x_positions = read_csv_value(folder_path)
        csv_val_dict = process_csv_value(folder_path)
        # y_ranges, expected_prob = get_y_range(csv_val_dict)

    # under 95%, the probability of people selecting our approach is within the range k/N
    number_of_observers = 15
    # Plot vertical lines
    PLOT = False # True False
    SAVE = False
    SHOW = False
    if PLOT:
        # x_positions = read_csv_value(folder_path)
        # print(f'x_positions {x_positions}') # x_positions
        csv_val_dict = process_csv_value(folder_path)
        y_ranges, expected_prob = get_y_range(csv_val_dict)
        print(f'y_ranges \n {y_ranges}')
        print(f'expected_prob \n {expected_prob}')

        plt.figure(figsize=(10, 6))
        for i, (bitrate, speed) in enumerate([(b, s) for b in bitrates for s in speeds]):
            y_min, y_max = y_ranges[speed][bitrates.index(bitrate)]
            prob = expected_prob[speed][bitrates.index(bitrate)]
            # print(f'prob {prob}')
            y_mid = (y_min + y_max) / 2
            y_error = (y_max - y_min) / 2
            # plt.vlines(i, y_min, y_max, label=speed if i % len(speeds) == 0 else "") # colors=colors[speed],
            plt.errorbar(i, y_mid, yerr=y_error, color=colors[speed], ecolor=colors[speed], \
                        elinewidth=5, capsize=8, label=speed if i % len(speeds) == 0 else "")
            plt.scatter(i, prob, color=colors[speed], marker='o', s=100, label=f'prob{prob}', )  # '^' is the triangle marker pointing upwards
            plt.text(i + 0.1, prob - 0.08, f'{round(prob, 2)}', color='grey', fontsize=18, ha='left', va='bottom')  # Adjust position with `ha` and `va`

        plt.axhline(0.5, color='lightgrey', linestyle='--', linewidth=2, label="y = 0.5")
        plt.text(0, 0.45, "p = 0.5", color='darkgrey', fontsize=15, ha='center')

        # # Draw vertical lines to separate bitrates
        # for i in range(3, len(x_labels), 3):  # Draw a line after every 3 labels (1Mbps, 2Mbps, etc.)
        #     plt.axvline(x=i - 0.5, color='darkgrey', linestyle='--', linewidth=1)
        
        # Draw vertical lines to separate bitrates and label them
        for idx, pos in enumerate(separator_positions):
            if pos != 12:
                plt.axvline(x=pos - 0.5, color='gray', linestyle='--', linewidth=1)
            # Add bitrate label beside the line
            if idx < len(bitrates):  # Ensure labels match the number of groups
                plt.text(pos - 0.5, 0, f'{bitrates_array[idx]}', fontsize=18, color='gray', ha='right', va='bottom')


        # Add x-axis labels and adjust layout
        plt.xticks(x_positions, x_labels, rotation=45, ha="right", fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(0, 1.05)
        plt.ylabel("Probability of selecting ours", fontsize=15)
        # plt.xlabel("Speeds and Bitrates")
        # plt.title("Vertical Lines for Speeds Across Bitrates")
        # plt.legend()
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plt.text(
            0.1, 0.17,  # Coordinates in figure-relative units
            f"N={number_of_observers}",  # Text to display
            fontsize=15,
            color="black",
            transform=plt.gcf().transFigure,  # Transform coordinates to figure-relative
            ha="left", va="bottom"  # Align text to bottom-left
        )

        now = datetime.now()
        plot_pth = now.strftime("%Y-%m-%d-%H_%M")
        if SAVE:
            plt.savefig(f"experiment-{plot_pth}.svg", dpi=300, bbox_inches='tight')
        if SHOW:
            print('show')
            plt.show()
