## Distributional Discrepancy: A Metric for Unconditional Text Generation

Code for: [Distributional Discrepancy: A Metric for Unconditional Text Generation](https://arxiv.org/abs/2005.01282) 

## Requirement
We suggest you run the platform under Python 3.6+ with following libs:
* **tensorflow>=1.12.0**
* tensorflow_hub>=0.7.0
* numpy 1.16.4
* scipy 1.3.1
* nltk 3.4.5
* colorama 0.4.1
* CUDA 7.5+ (Suggested for GPU speed up, not compulsory)    

Or just type `pip install -r requirements.txt` in your terminal.

## Get Started

```bash
git clone https://github.com/anonymous1100/Distributional-Discrepancy.git
cd Distributional-Discrepancy
# run with default setting
python main.py

#You can also change the model and data in main.py and then run main.py
```


## Structure
- `data` folder: emnlp_news and image_coco dataset
- `models` folder: Training codes for LSTM and GPT-2
- `utils` folder: code for some evaluation
- `experiments` folder: Experimental results
    - `experiments/experiment_name/tmp` folder: Generated samples
    - `experiments/experiment_name/output` folder: some output files
    - `experiments/experiment_name/ckpts` folder: Trained model
    - `experiments/experiment_name/summary` folder: tensorboard record

## Contact
For any questions, feel free to open an issue via github, or to send me an email at <br /> `1061185275@qq.com`. <br />
