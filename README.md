#
# Name_Entity_Recognition_Pytorch


## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.10 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```
### or
```bash
bash initial_setup.sh
```
### For torch installation
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```
### For tensorflow installation
```
pip install  tensorflow==2.10.1
```
### STEP 05- commit and push the changes to the remote repository

## For more information check docs/.*md files
