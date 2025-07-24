#!/bin/bash
# OpenCV Environment Setup for macOS
# This script sets up the environment variables needed for successful OpenCV compilation

echo "Setting up OpenCV environment variables for macOS..."

export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select --print-path)/Toolchains/XcodeDefault.xctoolchain/usr/lib/"
export LDFLAGS=-L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
export LIBCLANG_PATH="/Library/Developer/CommandLineTools/usr/lib"
export DYLD_LIBRARY_PATH="/Library/Developer/CommandLineTools/usr/lib:$DYLD_LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1:/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include"