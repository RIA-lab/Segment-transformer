import numpy as np
import yaml
from utils import load_weight
from models import load_model
import torch
import os
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import argparse


@dataclass
class InfernceSeq:
    accession: str
    sequence: str
    label: float = 0
    labels_low: float = 0
    labels_high: float = 0


def parse_fasta(fasta_file):
    """
    Parse a FASTA file and return a list of InfernceSeq objects.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        list: List of InfernceSeq objects with accession and sequence.
    """
    inference_data = []
    current_accession = None
    current_sequence = []

    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # If we have a previous sequence, save it
                if current_accession and current_sequence:
                    sequence = ''.join(current_sequence)
                    inference_data.append(InfernceSeq(accession=current_accession, sequence=sequence))
                # Start new sequence
                current_accession = line[1:].split()[0]  # Take first word after '>' as accession
                current_sequence = []
            else:
                current_sequence.append(line)

        # Don't forget the last sequence
        if current_accession and current_sequence:
            sequence = ''.join(current_sequence)
            inference_data.append(InfernceSeq(accession=current_accession, sequence=sequence))

    return inference_data

def cal_range(pred_min, pred_max, target_range=10):
    """
    Shrinks the range proportionally while keeping the ratio of pred_min and pred_max.

    Args:
        pred_min (float): The minimum predicted value.
        pred_max (float): The maximum predicted value.
        target_range (float): The desired range width.

    Returns:
        tuple: Adjusted (pred_min, pred_max).
    """
    # Compute the current range
    current_range = pred_max - pred_min
    if current_range <= target_range:
        return pred_min, pred_max  # No need to adjust if already within the range

    # Compute the scaling factor
    scale_factor = target_range / current_range

    # Shrink pred_min and pred_max proportionally
    pred_center = (pred_min + pred_max) / 2
    new_pred_min = pred_center - (pred_center - pred_min) * scale_factor
    new_pred_max = pred_center + (pred_max - pred_center) * scale_factor

    return new_pred_min, new_pred_max


def plot_temperature_prediction(pred_min, pred_max, pred, accession, save_path=None):
    """
    Plot a temperature profile with gradient background, predicted range bar, and stable temperature marker.

    Args:
        pred_min (float): Predicted minimum temperature (range start).
        pred_max (float): Predicted maximum temperature (range end).
        pred (float): Predicted stable temperature.
        accession (str, optional): Accession ID for title display.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(0, 125)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Hide standard axes

    # Plot color gradient background
    gradient = np.linspace(0, 1, 500).reshape(1, -1)
    ax.imshow(
        gradient,
        extent=[0, 125, 0.4, 0.6],  # y-range defines height = 0.2
        aspect='auto',
        cmap='coolwarm',  # blue -> red
        alpha=1
    )

    # Top and bottom border of the background bar
    ax.plot([0, 125], [0.4, 0.4], color='black', linewidth=1)
    ax.plot([0, 125], [0.6, 0.6], color='black', linewidth=1)

    # Tick marks and numeric labels
    for x in range(0, 130, 10):
        ax.plot([x, x], [0.395, 0.605], color='black', linewidth=1)
        ax.text(x, 0.375, f"{x}", ha='center', va='top', fontsize=9)

    # Predicted temperature range bar
    ax.barh(
        y=0.5,
        width=pred_max - pred_min,
        left=pred_min,
        height=0.2,
        color='#007b82',  # deep teal
        edgecolor='black',
        zorder=3
    )

    if pred != '-':
        ax.plot([pred, pred], [0.4, 0.6], color='red', linewidth=2, zorder=4)
        ax.text(pred, 0.61, f"{pred:.3f}°C", color='red', ha='center', va='bottom', fontsize=10)

    # Title
    title = f"Predicted Enzyme Temperature Profile ({accession})"
    ax.set_title(title, fontsize=12, pad=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/{accession}_pred.png', bbox_inches='tight', dpi=300)

    plt.show()


class InferenceModel:
    def __init__(self, weight_path):
        config_path = os.path.join(weight_path, 'model_config.yaml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        Model, Collator = load_model(config['name'])
        self.collate_fn = Collator(config['pretrain_model'])
        self.model = Model(config)
        load_weight(self.model, os.path.join(weight_path, 'model.safetensors'))
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.model.inference = True
        self.model.eval()

    def inference(self, data, visualize=True):
        """
        Perform inference on the provided data.

        Args:
            data (list): List of InfernceSeq objects.
            visualize (bool): Whether to visualize the results.

        Returns:
            np.ndarray: Predicted values.
        """
        return self._inference(data, visualize)


    def _inference(self, data, visualize=True):
        inputs = self.collate_fn(data)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        else:
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            # print(outputs)
        for idx, item in enumerate(data):
            inference_accession = item.accession
            inference_seq = item.sequence

            if not os.path.exists(f'inference_results/{inference_accession}'):
                os.mkdir(f'inference_results/{inference_accession}')
            with open(f'inference_results/{inference_accession}/prediction.txt', 'w') as file:
                file.write(f'accession: {item.accession}\n')
                file.write(f'sequence: {inference_seq}\n')
                for k, v in outputs.items():
                    if k == 'loss':
                        continue
                    elif k == 'pred_min':
                        pred_min = v[idx].cpu().numpy()
                    elif k == 'pred_max':
                        pred_max = v[idx].cpu().numpy()
                    elif k == 'pred':
                        pred = v[idx].cpu().numpy()
                    elif 'attn' in k:
                        v = v[idx].cpu().numpy().tolist()
                        v = np.repeat(v, 8)
                        v = v[:len(inference_seq)]
                        attn_weights = pd.DataFrame({'attn_weights': v})
                        attn_weights.to_csv(f'inference_results/{inference_accession}/{k}.csv', index=False)

                pred_min, pred_max = cal_range(pred_min, pred_max)
                file.write(f'pred_min: {pred_min}\n')
                file.write(f'pred_max: {pred_max}\n')
                file.write(f'pred: {pred}\n')
                if visualize:
                    plot_temperature_prediction(pred_min, pred_max, pred, item.accession, f'inference_results/{inference_accession}')

        return outputs.pred.cpu().numpy()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model inference.')
    parser.add_argument('--weight', type=str, required=True, help='Path to the model weights file')
    parser.add_argument('--file', type=str, required=True, help='File containing enzyme sequences')
    args = parser.parse_args()

    # Load model
    model = InferenceModel(args.weight)
    inference_data = parse_fasta(args.file)
    model.inference(inference_data)


