# Low-Light Image Enhancement

This directory contains scripts and configurations for the low-light image enhancement task using the Restormer model.

## Training

To train the model, run the following command:

```bash
python basicsr/train.py -opt LLIM/Options/LowLight_Restormer.yml
```

## Testing

To test the model, run the following command:

```bash
python LLIM/test.py -opt LLIM/Options/LowLight_Restormer.yml --weights <path_to_your_model.pth>
```
