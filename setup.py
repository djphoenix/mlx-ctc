# Copyright © 2024 Yury Popov (@djphoenix).

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx_ctc",
        version="0.0.1",
        description="C++ and Metal extensions for MLX CTC Loss",
        ext_modules=[extension.CMakeExtension("mlx_ctc._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_ctc"],
        package_data={"mlx_ctc": ["*.so", "*.dylib", "*.metallib"]},
        extras_require={"dev": []},
        zip_safe=False,
        python_requires=">=3.8",
    )
