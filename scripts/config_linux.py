#!/usr/bin/python3

"""
This script builds NovaPhy using the GCC compiler on Linux. 
It allows users to specify custom paths for GCC and CUDA 
installations, and optionally enables IPC support. The 
script sets up the necessary environment variables and CMake
 arguments before invoking pip to build the package.
"""

import argparse
import os
import sys
import subprocess
import shlex
from pathlib import Path
from enum import Enum
from typing import Optional

class BuildTarget(Enum):
    PY = 1
    WHEEL = 2
    CMAKE = 3

class HostToolChain(Enum):
    Unspecifyed = 0
    GCC = 1
    LLVM = 2

class StandardLibrary(Enum):
    Unspecifyed = 0
    GNU = 1
    LLVM = 2

# TODO IMPLEMENTED LATER, currently only support NVCC
# class CUDA_Compiler(Enum):
#     Unspecifyed = 0
#     NVCC = 1
#     CLANG_CUDA = 2

class build_config:
    def __init__(self):
        # Build direcotry

        self.build_dir = ""
        self.install_prefix = ""

        # Toolchain

        self.gcc_dir = ""
        self.llvm_dir = ""
        self.cuda_dir = ""
        self.compiler_launcher = ""
        self.use_libcpp = False

        # Target

        self.cmake_standalone = False
        self.build_wheel = False

        # Extra

        self.enable_ipc = False
        self.verbose_output = False

    def log(self, message: str) -> None:
        print(f"[build-linux-gcc] {message}")

    def debug(self, message: str) -> None:
        if self.verbose_output:
            self.log(message)

    def get_parser(self) -> argparse.ArgumentParser :
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--gcc-dir",
            type=str,
            default=None,
            help="Path to GCC installation directory",
            required=False
        )

        parser.add_argument(
            "--llvm-dir",
            type=str,
            default=None,
            help="Path to LLVM installation directory",
            required=False
        )

        parser.add_argument(
            "--cuda-dir",
            type=str,
            default=None,
            help="Path to CUDA installation directory",
            required=False
        )

        parser.add_argument(
            "--use-libcpp",
            action="store_true",
            help="Use libc++ instead of libstdc++ (only applicable when using LLVM toolchain)"
        )

        parser.add_argument(
            "--launcher",
            type=str,
            default=None,
            help="Program to launch compiler, e.g. ccache"
        )

        parser.add_argument(
            "--ipc",
            action="store_true",
            help="Enable IPC support"
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output for debugging"
        )

        parser.add_argument(
            "--build-dir",
            type=str,
            default=r"build/linux-gcc",
            help="Directory to store build files"
        )

        parser.add_argument(
            "--install-prefix",
            type=str,
            default=None,
            help="Installation prefix for CMake (only used with --cmake-standalone)"
        )

        parser.add_argument(
            "--cmake-standalone",
            action="store_true",
            help="Run cmake configuration step without scikit-build"
        )

        parser.add_argument(
            "--wheel",
            action="store_true",
            help="Build a wheel package instead of installing"
        )

        return parser

    def load_args(self) -> None:
        parser = self.get_parser()
        args = parser.parse_args()

        self.gcc_dir = args.gcc_dir
        self.llvm_dir = args.llvm_dir
        self.cuda_dir = args.cuda_dir
        self.compiler_launcher = args.launcher
        self.enable_ipc = args.ipc
        self.verbose_output = args.verbose
        self.build_dir = args.build_dir
        self.install_prefix = args.install_prefix
        self.cmake_standalone = args.cmake_standalone
        self.use_libcpp = args.use_libcpp
        self.build_wheel = args.wheel

        self.debug(
            "Parsed args: "
            f"build_dir={self.build_dir}, cmake_standalone={self.cmake_standalone}, "
            f"wheel={self.build_wheel}, ipc={self.enable_ipc}, "
            f"gcc_dir={self.gcc_dir}, llvm_dir={self.llvm_dir}, cuda_dir={self.cuda_dir}, "
            f"launcher={self.compiler_launcher}, use_libcpp={self.use_libcpp}"
        )

    def build_target(self) -> BuildTarget:
        if self.cmake_standalone:
            return BuildTarget.CMAKE
        elif self.build_wheel:
            return BuildTarget.WHEEL
        else:
            return BuildTarget.PY
        
    def host_toolchain(self) -> HostToolChain:
        if self.llvm_dir:
            return HostToolChain.LLVM
        elif self.gcc_dir:
            return HostToolChain.GCC
        else:
            return HostToolChain.Unspecifyed

    def host_compilers(self) -> Optional[tuple[Path, Path]] :
        if self.host_toolchain() == HostToolChain.Unspecifyed:
            return None
        elif self.host_toolchain() == HostToolChain.GCC:
            gcc_path = Path(self.gcc_dir) / "bin" / "gcc"
            gpp_path = Path(self.gcc_dir) / "bin" / "g++"
            return (gcc_path, gpp_path)
        elif self.host_toolchain() == HostToolChain.LLVM:
            clang_path = Path(self.llvm_dir) / "bin" / "clang"
            clangpp_path = Path(self.llvm_dir) / "bin" / "clang++"
            return (clang_path, clangpp_path)
        else:
            raise ValueError("Invalid host toolchain")

    def stdlib_include(self) -> Optional[Path] :
        if self.host_toolchain() == HostToolChain.Unspecifyed:
            return None
        elif self.host_toolchain() == HostToolChain.GCC:
            return Path(self.gcc_dir) / "include"
        elif self.host_toolchain() == HostToolChain.LLVM:
            if self.use_libcpp:
                return Path(self.llvm_dir) / "include" / "c++" / "v1"
            else:
                return Path(self.llvm_dir) / "include"
        else:
            raise ValueError("Invalid host toolchain")

    def cc_flags(self) -> list[str]:
        if self.host_toolchain() == HostToolChain.LLVM:
            flags = []
            if self.use_libcpp:
                flags.append("-stdlib=libc++")
            elif self.host_toolchain() == HostToolChain.GCC:
                flags.append("--gcc-toolchain=" + str(Path(self.gcc_dir)))
            return flags
        else:
            return []
    
    def cuda_compiler(self) -> Optional[Path] :
        if self.cuda_dir:
            return Path(self.cuda_dir) / "bin" / "nvcc"
        else:
            return None
    
    def nvcc_flags(self) -> list[str]:
        if self.cuda_dir and self.host_toolchain() != HostToolChain.Unspecifyed:
            cuda_include = Path(self.cuda_dir) / "include"
            stdlib_include = self.stdlib_include()
            if stdlib_include:
                return [f"--system-include={cuda_include},{stdlib_include}"]
            else:
                raise ValueError("Unexpected error: stdlib include path is None while host toolchain is specified")
        return []

    def setup_env(self) -> None :
        compilers = self.host_compilers()
        if compilers:
            cc_path, cxx_path = compilers
            os.environ["CC"] = str(cc_path)
            os.environ["CXX"] = str(cxx_path)
            self.debug(f"Set CC={cc_path}")
            self.debug(f"Set CXX={cxx_path}")
        else:
            self.debug("Using default system C/C++ compilers")

        cuda_compiler = self.cuda_compiler()
        if cuda_compiler:
            os.environ["CUDA_HOME"] = str(cuda_compiler.parent.parent)
            self.debug(f"Set CUDA_HOME={cuda_compiler.parent.parent}")
        else:
            self.debug("CUDA compiler not specified")

        if self.compiler_launcher:
            os.environ["CMAKE_CUDA_COMPILER_LAUNCHER"] = self.compiler_launcher
            os.environ["CMAKE_CXX_COMPILER_LAUNCHER"] = self.compiler_launcher
            os.environ["CMAKE_C_COMPILER_LAUNCHER"] = self.compiler_launcher
            self.debug(f"Set compiler launcher={self.compiler_launcher}")

    def cmake_args(self) -> list[str]:
        args = []
        if self.enable_ipc:
            args.append("--preset=ipc")
        else:
            args.append("--preset=default")

        if self.install_prefix and self.cmake_standalone:
            args.append(f"--install-prefix={self.install_prefix}")
        elif self.install_prefix and not self.cmake_standalone:
            print("Warning: --install-prefix is only used with --cmake-standalone. Ignoring it.")

        if self.build_wheel and self.cmake_standalone:
            print("Warning: --wheel option is not compatible with --cmake-standalone. Ignoring it.")
        
        compilers = self.host_compilers()
        if compilers:
            cc_path, cxx_path = compilers
            args.append(f"-DCMAKE_C_COMPILER={cc_path}")
            args.append(f"-DCMAKE_CXX_COMPILER={cxx_path}")

        cuda_compiler = self.cuda_compiler()
        if cuda_compiler:
            args.append(f"-DCMAKE_CUDA_COMPILER={cuda_compiler}")
            if compilers:
                _, cxx_path = compilers
                args.append(f"-DCMAKE_CUDA_HOST_COMPILER={cxx_path}")
                nvcc_flags = self.nvcc_flags()
                if nvcc_flags:
                    args.append(f"-DCMAKE_CUDA_FLAGS=\"{' '.join(nvcc_flags)}\"")

        if self.compiler_launcher:
            args.append(f"-DCMAKE_CUDA_COMPILER_LAUNCHER={self.compiler_launcher}")
            args.append(f"-DCMAKE_CXX_COMPILER_LAUNCHER={self.compiler_launcher}")
            args.append(f"-DCMAKE_C_COMPILER_LAUNCHER={self.compiler_launcher}")

        self.debug(f"Generated CMake args: {args}")
        return args




def main():
    cfg = build_config()
    cfg.load_args()
    cfg.setup_env()
    cmake_args = cfg.cmake_args()

    target = cfg.build_target()
    cfg.log(f"Build target: {target.name}")
    cfg.log(f"Build directory: {cfg.build_dir}")

    if target == BuildTarget.CMAKE:
        command = ["cmake", "-S", ".", "-B", cfg.build_dir, *cmake_args]
    elif target == BuildTarget.WHEEL:
        os.environ["CMAKE_ARGS"] = " ".join(cmake_args)
        cfg.debug(f"Set CMAKE_ARGS={os.environ['CMAKE_ARGS']}")
        command = [sys.executable, "-m", "pip", "wheel", ".", "-Cbuild-dir=" + cfg.build_dir]
    else:
        os.environ["CMAKE_ARGS"] = " ".join(cmake_args)
        cfg.debug(f"Set CMAKE_ARGS={os.environ['CMAKE_ARGS']}")
        command = [sys.executable, "-m", "pip", "install", ".", "-Cbuild-dir=" + cfg.build_dir]

    cfg.log(f"Running command: {shlex.join(command)}")
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
