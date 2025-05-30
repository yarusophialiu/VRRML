import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.stats import binom
from datetime import datetime
  

def read_csv_value(folder_path):
    result_list = []
    for file_name in os.listdir(folder_path):
        # print(f'file_name {file_name}')
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, nrows=10)
            # data = data.dropna()
            result_list.extend(data.iloc[:, [1, 2, 3]].values.tolist()) # bitrate, speed, score
    # print(f'result_list {len(result_list)} {result_list}\n')
    return result_list


def process_csv_value_no_speeds(folder_path):
    x_positions = read_csv_value(folder_path)
    print(f'x_positions {x_positions}')
    nested_dict = {}
    for bitrate, speed, score in x_positions:
        if bitrate not in nested_dict:
            nested_dict[bitrate] = []  # Initialize sub-dictionary for this bitrate
        # if speed not in nested_dict[bitrate]:
        #     nested_dict[bitrate][speed] = []  # Initialize an empty list for this speed
        nested_dict[bitrate].append(score)  # Append the score to the list for this speed

    print(f'nested_dict {nested_dict}\n')
    # Transform 0.4 → 0 and 0.6 → 1
    # nested_dict = {
    #     # v == 0.6 is max jod, 0.4 0.25 jod
    #     outer_k: {
    #         inner_k: [1 if v == 0.6 else 0 if v == 0.4 else v for v in inner_v]
    #         for inner_k, inner_v in outer_v.items()
    #     }
    #     for outer_k, outer_v in nested_dict.items()
    # }

    # print(f'nested_dict {nested_dict}')
    return nested_dict


def get_y_range(nested_dict):
    print(f'nested_dict {nested_dict}')
    y_range = {}
    expected_prob = {}
    for bitrate in bitrates:
        y_range[bitrate] = [] if bitrate not in y_range else y_range[bitrate]
        expected_prob[bitrate] = [] if bitrate not in expected_prob else expected_prob[bitrate]
        # for speed in speeds:
        # y_range = [] if speed not in y_range else y_range[speed]
        # expected_prob[speed] = [] if speed not in expected_prob else expected_prob[speed]
        print(f'===== bitrate {bitrate} =====')
        if bitrate in nested_dict:
            print(f'nested_dict[bitrate] {nested_dict[bitrate]}')
            p = sum(nested_dict[bitrate]) / len(nested_dict[bitrate])
            # expected_prob.append(p)
            k = binom.ppf(cumulative_threshold, N, p)
            # print(k/N)
            print(f'p {p}, k {k/N}')
            y_range[bitrate].append(tuple(k/N))
            expected_prob[bitrate].append(p)
        # else:
        #     y_range[bitrate].append((-1,-1))
        #     expected_prob[bitrate].append(-1)
    return y_range, expected_prob


def get_y_range_no_sparate_speeds(nested_dict):
    y_range = {}
    expected_prob = {}
    for bitrate in bitrates:
        for speed in speeds:
            y_range[speed] = [] if speed not in y_range else y_range[speed]
            expected_prob[speed] = [] if speed not in expected_prob else expected_prob[speed]
            print(f'===== bitrate {bitrate}, speed {speed} =====')
            if bitrate in nested_dict and speed in nested_dict[bitrate]:
                # print(f'nested_dict[bitrate][speed] {nested_dict[bitrate][speed]}')
                p = sum(nested_dict[bitrate][speed]) / len(nested_dict[bitrate][speed])
                # expected_prob.append(p)
                k = binom.ppf(cumulative_threshold, N, p)
                # print(k/N)
                print(f'p {p}, k {k/N}')
                y_range[speed].append(tuple(k/N))
                expected_prob[speed].append(p)
            else:
                y_range[speed].append((-1,-1))
                expected_prob[speed].append(-1)
    return y_range, expected_prob


def plot_separate_bitrate(folder_path, bitrates, speeds):
    # x_positions = read_csv_value(folder_path)
    csv_val_dict = process_csv_value_no_speeds(folder_path)
    y_ranges, expected_prob = get_y_range(csv_val_dict)

    print(f'y_ranges \n {y_ranges}')
    print(f'expected_prob \n {expected_prob}')

    plt.figure(figsize=(10, 6))
    for i, bitrate in enumerate(bitrates):
        print(f'bitrate {bitrate}')
        y_min, y_max = y_ranges[bitrate][0]
        prob = expected_prob[bitrate][0]
        print(f'prob {prob}')
        y_mid = (y_min + y_max) / 2
        y_error = (y_max - y_min) / 2
        # plt.vlines(i, y_min, y_max, label=speed if i % len(speeds) == 0 else "") # colors=colors[speed],
        plt.errorbar(i, y_mid, yerr=y_error, color=colors[3], ecolor=colors[3], \
                     elinewidth=5, capsize=8, label=3 if i % len(speeds) == 0 else "")
        plt.scatter(i, prob, color=colors[3], marker='o', s=100, label=f'prob{prob}', )  # '^' is the triangle marker pointing upwards
        plt.text(i + 0.0, prob - 0.08, f'{round(prob, 2)}', color='grey', fontsize=18, ha='left', va='bottom')  # Adjust position with `ha` and `va`

    plt.axhline(0.5, color='lightgrey', linestyle='--', linewidth=2, label="y = 0.5")
    plt.text(0, 0.45, "p = 0.5", color='darkgrey', fontsize=15, ha='center')
    x_labels = ['0.5Mbps', '1Mbps', '2Mbps', '3Mbps']
    x_positions = [0,1,2,3]
    
    
    # # Draw vertical lines to separate bitrates and label them
    # separator_positions = [1,2,3]
    # for idx, pos in enumerate(separator_positions):
    #     if pos != 0:
    #         plt.axvline(x=pos - 0.5, color='gray', linestyle='--', linewidth=1)
    #     # # Add bitrate label beside the line
    #     # if idx < len(bitrates):  # Ensure labels match the number of groups
    #     #     plt.text(pos - 0.5, 0, f'{bitrates_array[idx]}', fontsize=18, color='gray', ha='right', va='bottom')

    # Add x-axis labels and adjust layout
    ax = plt.gca()
    import matplotlib.transforms as mtransforms

    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
        offset = mtransforms.ScaledTranslation(0.5, -0.05, ax.figure.dpi_scale_trans)  # shift right 5 points
        label.set_transform(label.get_transform() + offset)
    # # Move x-tick labels slightly to the right
    # for label in ax.get_xticklabels():
    #     label.set_x(label.get_position()[0] + 1.05)

    # # Move y-tick labels slightly upward
    # for label in ax.get_yticklabels():
    #     label.set_y(label.get_position()[1] + 0.)

    plt.xticks(x_positions, x_labels, ha="right", fontsize=15)
    # plt.xticks(x_ticks, x_labels, fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.ylim(0, 1.05)
    plt.yticks([i * 0.2 for i in range(1, 6)], fontsize=15)
    # plt.ylim(0, 1.05)
    plt.ylabel("Probability of selecting maxjod", fontsize=15)
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
        plt.savefig(f"experiment_results/experiment-drpjod-speeds{plot_pth}.svg", dpi=300, bbox_inches='tight')
    if SHOW:
        plt.show()



def plot_without_speeds(folder_path, bitrates, speeds):
    csv_val_dict = process_csv_value_no_speeds(folder_path)
    y_ranges, expected_prob = get_y_range(csv_val_dict)

    print(f'y_ranges \n {y_ranges}')
    print(f'expected_prob \n {expected_prob}')




if __name__ == "__main__":
    # Data definition
    # bitrates = ['1', '2', '4', '8']
    bitrates = [500, 1000, 1500, 2000]
    speeds = [1.0, 2.0, 3.0] # ['v1', 'v2', 'v3']
    speeds_dict = {1: 'Slow', 2: 'Medium', 3: 'Fast'}
    bitrates_dict = {500: 1, 1000: 2, 1500: 3, 2000: 4}
    bitrates_array = ['0.5Mbps ', '1Mbps ', '1.5Mbps ', '2Mbps']
    # separator_positions = [3, 6, 9, 12]  # Positions where separators are drawn

    x_labels = [f"{speeds_dict[speed]}" for bitrate in bitrates for speed in speeds]
    cumulative_threshold = [0.025, 0.975] # Cumulative probability threshold, 95% confidence interval
    # N = 3 * 3 * 4 # Number of trials: 3 test scenes, 3 speeds, 4 bitrates
    N = 10
    folder_path = 'JOD25'

    colors = {
        1: "gold",
        2: "deepskyblue",
        3: "salmon"
    }

    PROCESS_CSV = True# True False
    if PROCESS_CSV:
        # x_positions = read_csv_value(folder_path)
        csv_val_dict = process_csv_value_no_speeds(folder_path)
        y_ranges, expected_prob = get_y_range(csv_val_dict)

    # under 95%, the probability of people selecting our approach is within the range k/N
    number_of_observers = 10
    PLOT_SEPARATE_SPEEDS = False # True False
    PLOT_WITHOUT_SPEEDS = True # True False
    SAVE = False
    SHOW = True
    # if PLOT_SEPARATE_SPEEDS:
    #     plot_separate_speeds(folder_path, bitrates, speeds)

    if PLOT_WITHOUT_SPEEDS:
        plot_separate_bitrate(folder_path, bitrates, speeds)
