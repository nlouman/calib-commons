from setuptools import setup, find_packages

setup(
    name="calib_commons",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # Ici, ajoutez les dépendances nécessaires, si il y en a.
    ],
    entry_points={
        "console_scripts": [
            # 'name-of-command = package.module:function'
            "calibrate-intrinsics = calib_commons.scripts.internal_calibration:main",
            "calib-eval = calib_commons.scripts.run_eval_calib:main",
            "center-dslr = calib_commons.scripts.center_to_dslr:main",
        ],
    },
)
