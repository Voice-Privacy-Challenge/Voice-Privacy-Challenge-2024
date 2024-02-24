#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)

home=$PWD
\rm env.sh 2> /dev/null || true
touch env.sh

# VENV install dir
venv_dir=$PWD/venv
export MAMBA_ROOT_PREFIX=".micromamba"  # Local install of micromamba (where the libs/bin will be cached)
mamba_bin="$MAMBA_ROOT_PREFIX/micromamba"

### VERSION

MAMBA_VERSION=1.5.1-0

CUDA_VERSION=11.7
TORCH_VERSION=2.0.1

MAMBA_PACKAGES_TO_INSTALL="sshpass OpenSSH sox libflac tar libacl inotify-tools git-lfs ffmpeg wget make cmake ncurses ninja python=3.10 nvtop automake libtool gxx=12.3.0 gcc=12.3.0 python-sounddevice"

mark=.done-venv
if [ ! -f $mark ]; then
  echo " == Making virtual environment =="
  if [ ! -f "$mamba_bin" ]; then
    echo "Downloading micromamba"
    mkdir -p "$MAMBA_ROOT_PREFIX"
    curl -sS -L "https://github.com/mamba-org/micromamba-releases/releases/download/$MAMBA_VERSION/micromamba-linux-64" > "$mamba_bin"
    chmod +x "$mamba_bin"
  fi
  [ -d $venv_dir ] && yes | rm -rf $venv_dir

  echo "Micromamba version:"
  "$mamba_bin" --version

  "$mamba_bin" create -y --prefix "$venv_dir"

  echo "Installing conda dependencies"
  "$mamba_bin" install -y --prefix "$venv_dir" -c conda-forge $MAMBA_PACKAGES_TO_INSTALL || exit 1
  "$venv_dir/bin/python" --version || exit 1

  touch $mark
fi


if [ -e "$venv_dir" ]; then export PATH="$venv_dir/bin:$PATH"; fi

# Hook Micromamba into the script's subshell (this only lasts for as long as the # script is running)
echo "#!/bin/bash" >> env.sh
echo "eval \"\$($mamba_bin shell hook --shell=bash)\"" >> env.sh
echo "micromamba activate $venv_dir" >> env.sh
echo "export LD_LIBRARY_PATH=$venv_dir/lib/:$LD_LIBRARY_PATH" >> env.sh
echo "alias conda=micromamba" >> env.sh
echo "export PIP_REQUIRE_VIRTUALENV=false" >> env.sh
source ./env.sh


mark=.done-cuda
if [ ! -f $mark ]; then
  echo " == Installing cuda =="
  micromamba install -y --prefix "$venv_dir" -c "nvidia/label/cuda-${CUDA_VERSION}.0" cuda-toolkit || exit 1
  "$venv_dir/bin/nvcc" --version || exit 1
  touch $mark
fi

CUDAROOT=$venv_dir
echo "export CUDAROOT=$CUDAROOT" >> env.sh
source ./env.sh


cuda_version_without_dot=$(echo $CUDA_VERSION | xargs | sed 's/\.//')
mark=.done-pytorch
if [ ! -f $mark ]; then
  echo " == Installing pytorch $TORCH_VERSION for cuda $CUDA_VERSION =="
  version="==$TORCH_VERSION+cu$cuda_version_without_dot"
  echo -e "\npip3 install torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_without_dot\n"
  pip3 install torch$version torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/${nightly}cu$cuda_version_without_dot \
    || { echo "Failed to find pytorch $TORCH_VERSION for cuda '$CUDA_VERSION', use specify other torch/cuda version (with variables in install.sh script)"  ; exit 1; }
  python3 -c "import torch; print('Torch version:', torch.__version__)" || exit 1
  touch $mark
fi


mark=.done-python-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  pip3 install -r requirements.txt  || exit 1
  touch $mark
fi

echo " == Everything got installed successfully =="
