# Build NovaPhy

[TOC]

## Build on Windows with MSVC

TODO

## Build on Linux with GCC/Clang

### Basic

NovaPhy without IPC support just needs the following prerequisites (as mentioned in [README.md](../README.md#setup)):

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- [vcpkg](https://vcpkg.io/) installed. *Optional*
- C++17 compiler
  - MSVC 2019+
  - GCC 9+
  - Clang 10+

```bash
# Create conda environment
conda env create -f environment.yml
conda activate novaphy

# Install NovaPhy
CMAKE_ARGS="--preset=default" pip install -e .

# Then you can access NovaPhy from Python in the virtual environment.
python
```

### IPC support

NovaPhy relies on [libuipc](https://github.com/spiriMirror/libuipc) for IPC support.
It requires C++20 features, and its upstream dependencies depend on some non-standard features.
Thus, the prerequisites are more specific.

- [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- [vcpkg](https://vcpkg.io/) installed.
- C++20 compiler
  - MSVC 2019+
  - GCC 11, 12, 13
  - Clang 10+
- CUDA 12.4

```bash
conda activate novaphy

# Before this step, make sure your environment variables are set, such as VCPKG_ROOT.
git submodule update --init --recursive # Update submodule directory.
CMAKE_ARGS="--preset=ipc" pip install -e .
```

### Troubleshooting

If you use a compiler not listed above, it may fail with the default configuration.
Here are some common reasons for crashes with specific compilers:

| Compiler | Reason | Status |
|:---:|:---|:---:|
| `gcc-9` | **libuipc** needs `<span>`, which was introduced in GCC 10. | Unfixable |
| `gcc-10` | The PSTL in `libstdc++ 10` isn't compatible with `onetbb` in the current baseline. (See also [vcpkg.json](../vcpkg.json).) | Unfixable |
| `gcc-15` | `urdfdom` expects a non-standard import for `uint32_t` which is removed from `libstdc++`. | Fixed [^1] |
| `gcc-14, gcc-15` | `gcc>13` is not compatible with `nvcc-12` | Unfixable |
| `nvcc-13` | **muda** expects non-strict dependent name resolution. | TODO |

[^1] : The issue has been fixed by upstream maintainers with a patch. However, the patch hasn't been included in the current baseline (2025.07.25). Adding `-include cstdint` to your `CXXFLAGS` environment variable should resolve the problem.

> [!note] Clang with libstdc++
> Clang uses `libstdc++` as the default standard library. Any crash caused by `libstdc++` also affects Clang.

Compilers are described by name and major version, such as `gcc-9`. For each major version, only one version is tested (✅ marks compatible compilers):

- GCC 9.5.0
- GCC 10.5.0
- ✅ GCC 11.5.0
- ✅ GCC 12.5.0
- ✅ GCC 13.4.0
- GCC 14.3.0
- GCC 15.2.1
- ✅ nvcc-12: Cuda compilation tools, release 12.4, V12.4.131
- nvcc-13: Cuda compilation tools, release 13.1, V13.1.115

## Components packaging and install

NovaPhy has two components overall:

- A Python binding as frontend
- A C++ library as backend

The frontend and backend can be built, packaged, and installed independently.

Two methods are provided for handling the backend's largest dependency, `libuipc`: bundle and standalone. The former means `libuipc` will be bundled into the backend component. NovaPhy will not use `libuipc` from the system or virtual environment. The latter means `libuipc` will be imported from the environment, and the binary target of `libuipc` will not be included in the package. The `libuipc` in this context is refered to its C++ library. Its python binding would never be included in the wheel package of NovaPhy.

### Package with python build tools

Using python build with `scikit-build-core` will build and packaging two part of NovaPhy together into a wheel package. It is convenient for python developer. To control the processing mode of `libuipc`, the CMake options `NOVAPHY_BUNDLE_UIPC` is provided. Set it to ON (default) will bundle `libuipc` to the `wheel` package. Otherwise, user need to gurantee `NovaPhy` could find the `libuipc` and `cuda` runtime.

```bash
pip install build
python -m build

# Or, standalone libuipc
CMKAE_ARGS="-DNOVAPHY_BUNDLE_UIPC=OFF" python -m build
```

### Standalone CMake

If you want to use NovaPhy in C++ program, you can build NovaPhy as a normal CMake project. You can install NovaPhy to your machine by CMake (as show following), or use `add_subdirectory(/path/to/NovaPhy)` directly.

```bash
mkdir build
cd build
cmake -S .. -B . --preset=default --install-prefix=/path/to/installation
# or
cmake -S .. -B . --preset=ipc ... # to enable ipc support

# build
cmake --build .

# install
# TODO install for CMake standalone build.
```

It will build both the backend and the Python binding. However, only the backend can be packaged and installed by CMake. An independent Python binding will be generated in `python/` in the build directory. Then you can use native Python build tools to package the frontend. An additional packaging script would help handle this case.

TODO: Packaging script