echo [$(DATE)]: "START"
echo [$(DATE)]: "creating environment"
conda create --prefix ./env python=3.10 -y
echo [$(date)]: "activate environment"
source activate ./env
conda activate ./env
echo [$(date)]: "create folder and file structure"

for dir in components config constants entity exception_and_logger pipeline utils
do
    echo [$(date)]: "creating" ner/$dir
    mkdir -p ner/$dir
    echo [$(date)]: "Creating __init__.py inside above folders"
    touch ner/__init__.py ner/$dir/__init__.py 
done
echo [$(date)]: "installing pytorch"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

echo [$(date)]: "installing tensorflow"
pip install  tensorflow==2.10.1
pip install  protobuf==3.20.0

echo [$(date)]: "install requirements"
pip install -r requirements.txt

# pip install torch --extra-index-url https://download.pytorch.org/whl/cu113 -q
echo [$(date)]: "END"
