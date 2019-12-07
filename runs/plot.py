#!/usr/bin/env python
from ipdb import set_trace as st
from os import path
import csv
import matplotlib.pyplot as plt

filename = {
        'test accuracy': 'run-validation-tag-epoch_accuracy.csv',
        'training loss': 'run-train-tag-epoch_loss.csv'}

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
for axes_row, metric in zip(axes, ['test accuracy', 'training loss']):
    for axis, optimizer in zip(axes_row, ['RAdam', 'Adam', 'SGD']):
        if metric == 'test accuracy':
            axis.set_title(optimizer)
        else:
            axis.set_xlabel('Epoch')
        axis.set_xlim(left=0, right=163)
        axis.set_ylabel(metric)
        axis.set_ylim(top=1.0, bottom=0.0)
        axis.grid(linewidth=0.1)
        for lr in ['0.1', '0.03', '0.01', '0.003', '0.001']:
            filepath = "{optimizer} lr={lr}/{filename}".format(
                    optimizer=optimizer,
                    lr=lr,
                    filename=filename[metric])
            if not path.exists(filepath): # Del this later
                continue
            x_vals, y_vals = [], []
            # Read lines of CSV file
            with open(filepath) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    if csv_reader.line_num == 1:
                        continue
                    x_vals.append(int(line[1]))
                    y_vals.append(float(line[2]))
            # Plot
            axis.plot(x_vals, y_vals, label=lr)
            axis.legend(title='Initial Learning Rate')
plt.tight_layout()
plt.savefig('resnet.pdf')
