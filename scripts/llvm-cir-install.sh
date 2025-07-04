#!/usr/bin/env bash

REPO_URL="https://github.com/llvm/clangir.git"

# Check if -y arg is passed
if [[ "$1" == "-y" ]]; then
  AUTO_CONFIRM=true
else
  AUTO_CONFIRM=false
fi

# Check if LLVM_CIR_COMMIT_HASH environment variable is set
if [ -z "$LLVM_CIR_COMMIT_HASH" ]; then
  echo "Error: LLVM_CIR_COMMIT_HASH environment variable is not set."
  exit 1
fi

# Check if the $LLVM_SOURCE_PATH environment variable is set
if [ -z "$LLVM_SOURCE_PATH" ]; then
  echo "Error: LLVM_SOURCE_PATH environment variable is not set."
  exit 1
fi

# Check if the $LLVM_SOURCE_PATH exists
if [ -d "$LLVM_SOURCE_PATH" ]; then
  # Ask for user confirmation to remove the directory
  if [ "$AUTO_CONFIRM" = false ]; then
    read -r -p "Directory $LLVM_SOURCE_PTH already exists. Do you want to remove it? (y/N): " confirm
  else
    confirm="Y" # Auto-confirm if -y is passed
  fi
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Removing existing directory $LLVM_SOURCE_PATH..."
    rm -rf "$LLVM_SOURCE_PATH"
  else
    echo "Build cancelled. Directory $LLVM_SOURCE_PATH will not be removed."
    exit 0
  fi
fi
echo "Using source directory: $LLVM_SOURCE_PATH"

# Create the directory for LLVM CIR
mkdir -p "$LLVM_SOURCE_PATH"

# Move into the directory
cd "$LLVM_SOURCE_PATH" || exit 1

# Clone the commit from the repository
git init
git remote add origin "$REPO_URL"
git fetch --depth=1 origin "$LLVM_CIR_COMMIT_HASH"
git checkout FETCH_HEAD

# Check that all LLVM env vars are set and paths are valid
if [ -z "$LLVM_BUILD_PATH" ]; then
  echo "Error: LLVM_BUILD_PATH environment variable is not set."
  exit 1
fi
if [ ! -d "$LLVM_BUILD_PATH" ]; then
  echo "LLVM_BUILD_PATH does not exist. Creating it..."
  mkdir -p "$LLVM_BUILD_PATH"
else
  if [ "$AUTO_CONFIRM" = false ]; then
    read -r -p "Build directory $LLVM_BUILD_PATH already exists. Do you want to clean it? (y/N): " confirm
  else
    confirm="Y" # Auto-confirm if -y is passed
  fi
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cleaning existing build directory $LLVM_BUILD_PATH..."
    rm -rf "${LLVM_BUILD_PATH:?}"/*
  fi
fi
echo "Using build directory: $LLVM_BUILD_PATH"

if [ -z "$LLVM_INSTALL_PATH" ]; then
  echo "Error: LLVM_INSTALL_PATH environment variable is not set."
  exit 1
fi
if [ ! -d "$LLVM_INSTALL_PATH" ]; then
  echo "LLVM_INSTALL_PATH does not exist. Creating it..."
  mkdir -p "$LLVM_INSTALL_PATH"
else
  read -r -p "Install directory $LLVM_INSTALL_PATH already exists. Do you want to clean it? (y/N): " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cleaning existing install directory $LLVM_INSTALL_PATH..."
    rm -rf "${LLVM_INSTALL_PATH:?}"/*
  fi
fi
echo "Using install directory: $LLVM_INSTALL_PATH"

# Check for C compiler
CLANG=$(which clang)
echo "Using C compiler: $CLANG"
echo "Using C++ compiler: ${CLANG}++"

# Ask confirmation to proceed with the build
if [ "$AUTO_CONFIRM" = false ]; then
  read -r -p "Do you want to proceed with the build? (y/N): " confirm
else
  confirm="Y" # Auto-confirm if -y is passed
fi
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Build cancelled."
  exit 0
fi

# Configure the build
echo "Configuring the build in $LLVM_BUILD_PATH..."
cd "$LLVM_BUILD_PATH" || exit 1
cmake -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PATH" \
  -DCMAKE_CXX_COMPILER="${CLANG}++" \
  -DCMAKE_C_COMPILER="$CLANG" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_ENABLE_LLD=ON \
  -DCLANG_ENABLE_CIR=ON \
  "${LLVM_SOURCE_PATH}/llvm"

# Build
echo "Building LLVM CIR..."
ninja install

echo "LLVM CIR has been built and installed successfully."
