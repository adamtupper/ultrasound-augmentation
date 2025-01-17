# Dataset of B-mode fatty liver ultrasound images

The Dataset of B-mode fatty liver ultrasound images contains 550 B-mode ultrasound images from 55 obese patients. Each image has a resolution of 434 x 636 pixels.

([Paper](https://link.springer.com/article/10.1007/s11548-018-1843-2), [Dataset](https://zenodo.org/records/1009146))

## Pre-processing

```mermaid
flowchart TD
A["Load matrix file (.mat)"]
B["Extract frame sequences, patient IDs, labels, and steatosis values"]
C[Save each frame as a PNG file]
D["Group images by patient and split into training, validation, and test data using a 70:10:20 split"]
A-->B
B-->C
C-->D
```
