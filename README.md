# :ninja:Train pipeline for recognizer:fire:

## Prepare environment

1. Install requirements.txt
```bash
/usr/bin/python3 -m pip install -r requirements.txt
```
2. Setup variables and creds for mlflow and minio connection
```bash
export MLFLOW_TRACKING_URI=http://10.66.100.20:5000
export MLFLOW_S3_ENDPOINT_URL=http://10.66.100.20:9000
touch ~/.aws/credentials
cat <<EOF > ~/.aws/credentials
[default]
aws_access_key_id=minio
aws_secret_access_key=minio123
EOF
```

## Train

### 1. Specify/check config file
```train_config.yml``` file contains all hyperparameters to specify for train pipeline. Check main parts:
```yml
global:
  experiment_name: mlflow experiment name (e.g. country name)
  run_name: give brief name for your run (e.g. 100k_malaysia_europe)
  run_id: pass only if you want to continue specific run (get run id from mlflow UI)
  distributed: boolean[True/False]
  epochs: int
  checkpoint: path where previous weights are kept and new will be saved(non-existent folder will be automatically created)
  gpu: specify for single GPU launch/pass any value
  alphabet: string.digits+string.ascii_lowercase (e.g. only operations which can be processed with eval()

model:
  type: extended/any for base
  ---
  classification_regions: leave empty for no classification head, either way provide regions in list

---

train_data:
  csv_files: ['train1.csv'] only in list
```
### 2. Simply run pipeline
```bash
/usr/bin/python3 train.py
```
:warning:Warning:warning:: If mlflow host is different than set, modify tracking uri and train.py:
```python
mlflow.set_tracking_uri("http://10.66.100.20:5000")
```
## Test

### Check config and run
```test_config.yml``` is config file for test pipeline
```yml
images: csv file/folder/img path
evaluate: boolean[True/False] (if you have labels pass True)
output_folder: directory to save results
model:
  path: folder where model weights is located/model path
  ---
---
threshold: for calculating accuracy
---
```
Run test pipeline
```bash
/usr/bin/python3 test.py
```
