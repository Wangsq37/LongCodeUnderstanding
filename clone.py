# -*- coding: utf-8 -*-

import os
import subprocess

# 合并的Python科学计算库及其 GitHub 仓库 clone 链接
# 选择标准：
# 1. 复杂的跨文件依赖关系
# 2. 良好的环境配置（setup.py/pyproject.toml）
# 3. 数据计算、科学计算、并行计算相关
# 4. 活跃的开发和维护
# 5. 丰富的测试用例
# 6. 专业的科学计算应用场景

merged_repos = {
    # 核心数值计算库
    # "numpy": "https://github.com/numpy/numpy.git",
    # "scipy": "https://github.com/scipy/scipy.git",
    # "pandas": "https://github.com/pandas-dev/pandas.git",
    # "matplotlib": "https://github.com/matplotlib/matplotlib.git",
    # "sympy": "https://github.com/sympy/sympy.git",
    
    # 机器学习和深度学习
    # "scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    # "pytorch": "https://github.com/pytorch/pytorch.git",
    # "tensorflow": "https://github.com/tensorflow/tensorflow.git",
    # "jax": "https://github.com/google/jax.git",
    # "keras": "https://github.com/keras-team/keras.git",
    # "transformers": "https://github.com/huggingface/transformers.git",
    # "torchvision": "https://github.com/pytorch/vision.git",
    # "torchaudio": "https://github.com/pytorch/audio.git",
    # "autograd": "https://github.com/HIPS/autograd.git",
    # "theano": "https://github.com/Theano/Theano.git",
    
    # 科学计算和物理模拟
    "astropy": "https://github.com/astropy/astropy.git",
    "biopython": "https://github.com/biopython/biopython.git",
    "fenics": "https://github.com/FEniCS/dolfinx.git",
    "firedrake": "https://github.com/firedrakeproject/firedrake.git",
    "dealii": "https://github.com/dealii/dealii.git",
    "sfe": "https://github.com/kinnala/scikit-fem.git",
    "sfepy": "https://github.com/sfepy/sfepy.git",
    "fluidfft": "https://github.com/fluiddyn/fluidfft.git",
    "ScientificPython": "https://github.com/ScientificPython/ScientificPython.git",
    "fluids": "https://github.com/CalebBell/fluids.git",
    "cantera": "https://github.com/Cantera/cantera.git",
    "openfoam": "https://github.com/OpenFOAM/OpenFOAM.git",
    
    # 量子计算和物理模拟
    "qutip": "https://github.com/qutip/qutip.git",
    "cirq": "https://github.com/quantumlib/Cirq.git",
    "qiskit": "https://github.com/Qiskit/qiskit.git",
    "pennylane": "https://github.com/PennyLaneAI/pennylane.git",
    "projectq": "https://github.com/ProjectQ-Framework/ProjectQ.git",
    
    # 分子动力学和化学
    "ase": "https://gitlab.com/ase/ase.git",
    "pymatgen": "https://github.com/materialsproject/pymatgen.git",
    "phonopy": "https://github.com/phonopy/phonopy.git",
    "wannier90": "https://github.com/wannier-developers/wannier90.git",
    "mdanalysis": "https://github.com/MDAnalysis/mdanalysis.git",
    "rdkit": "https://github.com/rdkit/rdkit.git",
    "openmm": "https://github.com/openmm/openmm.git",
    "mdtraj": "https://github.com/mdtraj/mdtraj.git",
    
    # 天体物理和宇宙学
    "gadget": "https://gitlab.mpcdf.mpg.de/vrs/gadget4.git",
    "yt": "https://github.com/yt-project/yt.git",
    "halotools": "https://github.com/astropy/halotools.git",
    "21cmfast": "https://github.com/21cmfast/21cmFAST.git",
    
    # 地球科学
    "obspy": "https://github.com/obspy/obspy.git",
    "pyrocko": "https://git.pyrocko.org/pyrocko/pyrocko.git",
    "climlab": "https://github.com/brian-rose/climlab.git",
    "iris": "https://github.com/SciTools/iris.git",
    
    # 生物信息学和基因组学
    "pysam": "https://github.com/pysam-developers/pysam.git",
    "htseq": "https://github.com/htseq/htseq.git",
    "scanpy": "https://github.com/theislab/scanpy.git",
    
    # 并行计算和GPU加速
    "cupy": "https://github.com/cupy/cupy.git",
    "numba": "https://github.com/numba/numba.git",
    "pycuda": "https://github.com/inducer/pycuda.git",
    "pyopencl": "https://github.com/inducer/pyopencl.git",
    "pyfftw": "https://github.com/pyFFTW/pyFFTW.git",
    "pythran": "https://github.com/serge-sans-paille/pythran.git",
    "dask": "https://github.com/dask/dask.git",
    "ray": "https://github.com/ray-project/ray.git",
    "joblib": "https://github.com/joblib/joblib.git",
    
    # 数据分析和可视化
    "seaborn": "https://github.com/mwaskom/seaborn.git",
    "plotly": "https://github.com/plotly/plotly.py.git",
    "bokeh": "https://github.com/bokeh/bokeh.git",
    "geopandas": "https://github.com/geopandas/geopandas.git",
    "xarray": "https://github.com/pydata/xarray.git",
    "vaex": "https://github.com/vaexio/vaex.git",
    "holoviews": "https://github.com/holoviz/holoviews.git",
    "datashader": "https://github.com/holoviz/datashader.git",
    
    # 图像处理和计算机视觉
    "opencv": "https://github.com/opencv/opencv.git",
    "scikit-image": "https://github.com/scikit-image/scikit-image.git",
    "pillow": "https://github.com/python-pillow/Pillow.git",
    "imageio": "https://github.com/imageio/imageio.git",
    "albumentations": "https://github.com/albumentations-team/albumentations.git",
    "mahotas": "https://github.com/luispedro/mahotas.git",
    
    # 信号处理和音频
    "librosa": "https://github.com/librosa/librosa.git",
    "pykalman": "https://github.com/pykalman/pykalman.git",
    "pywavelets": "https://github.com/PyWavelets/pywt.git",
    
    # 优化和数值方法
    "nlopt": "https://github.com/stevengj/nlopt.git",
    "cvxpy": "https://github.com/cvxpy/cvxpy.git",
    "pulp": "https://github.com/coin-or/pulp.git",
    "pyomo": "https://github.com/Pyomo/pyomo.git",
    "openmdao": "https://github.com/OpenMDAO/OpenMDAO.git",
    "pydoe": "https://github.com/tisimst/pyDOE.git",
    
    # 统计和概率
    "statsmodels": "https://github.com/statsmodels/statsmodels.git",
    "pingouin": "https://github.com/raphaelvallat/pingouin.git",
    "arviz": "https://github.com/arviz-devs/arviz.git",
    "pymc": "https://github.com/pymc-devs/pymc.git",
    "emcee": "https://github.com/dfm/emcee.git",
    "lmfit": "https://github.com/lmfit/lmfit-py.git",
    "chaospy": "https://github.com/jonathf/chaospy.git",
    "bambi": "https://github.com/bambinos/bambi.git",
    
    # 网络和图形
    "networkx": "https://github.com/networkx/networkx.git",
    "igraph": "https://github.com/igraph/python-igraph.git",
    "graph-tool": "https://git.skewed.de/count0/graph-tool.git",
    "snap": "https://github.com/snap-stanford/snap.git",
    "stellargraph": "https://github.com/stellargraph/stellargraph.git",
    
    # 化学和分子模拟
    "mdanalysis": "https://github.com/MDAnalysis/mdanalysis.git",
    "rdkit": "https://github.com/rdkit/rdkit.git",
    "openmm": "https://github.com/openmm/openmm.git",
    "mdtraj": "https://github.com/mdtraj/mdtraj.git",
    
    # 地理和空间数据
    "rasterio": "https://github.com/rasterio/rasterio.git",
    "fiona": "https://github.com/Toblerity/Fiona.git",
    "shapely": "https://github.com/shapely/shapely.git",
    "pyproj": "https://github.com/pyproj4/pyproj.git",
    
    # 时间序列
    "prophet": "https://github.com/facebook/prophet.git",
    "arch": "https://github.com/bashtage/arch.git",
    "pyflux": "https://github.com/RJT1990/pyflux.git",
    "tslearn": "https://github.com/tslearn-team/tslearn.git",
    
    # 其他重要库
    "cython": "https://github.com/cython/cython.git",
    "pytensor": "https://github.com/pymc-devs/pytensor.git",
    "sagemath": "https://github.com/sagemath/sage.git",
    "zarr": "https://github.com/zarr-developers/zarr-python.git",
    "h5py": "https://github.com/h5py/h5py.git",
    "tables": "https://github.com/PyTables/PyTables.git",
    "netcdf4": "https://github.com/Unidata/netcdf4-python.git",
}

# 目标目录
target_dir = "python_repos"
os.makedirs(target_dir, exist_ok=True)

# 记录文件路径
record_file = os.path.join(target_dir, "python_repos.txt")

print("=" * 80)
print("开始克隆Python科学计算仓库")
print("选择标准：复杂跨文件依赖、良好环境配置、数据计算相关、专业应用场景")
print("=" * 80)

success_count = 0
skip_count = 0
fail_count = 0

with open(record_file, "w", encoding="utf-8") as f:
    for name, url in merged_repos.items():
        repo_path = os.path.join(target_dir, name)
        if os.path.exists(repo_path):
            print(f"[跳过] {name} 已存在")
            skip_count += 1
            continue
            
        print(f"[正在克隆] {name}...")
        try:
            result = subprocess.run(
                ["git", "clone", url, repo_path], 
                capture_output=True, 
                text=True, 
                timeout=900  # 15分钟超时（大仓库需要更长时间）
            )
            
            if result.returncode == 0:
                f.write(name + "\n")
                print(f"[成功] {name}")
                success_count += 1
            else:
                print(f"[失败] {name} - {result.stderr}")
                fail_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"[超时] {name} - 克隆超时")
            fail_count += 1
        except Exception as e:
            print(f"[错误] {name} - {str(e)}")
            fail_count += 1

print("=" * 80)
print("克隆完成统计:")
print(f"成功: {success_count}")
print(f"跳过: {skip_count}")
print(f"失败: {fail_count}")
print(f"总计: {len(merged_repos)}")
print(f"记录已保存到: {record_file}")
print("=" * 80)

# 生成仓库分类报告
print("\n仓库分类报告:")
print("-" * 50)
categories = {
    "核心数值计算": ["numpy", "scipy", "pandas", "matplotlib", "sympy"],
    "机器学习": ["scikit-learn", "pytorch", "tensorflow", "jax", "keras", "transformers", "autograd", "theano"],
    "量子计算": ["qutip", "cirq", "qiskit", "pennylane", "projectq"],
    "数值方法": ["fenics", "firedrake", "dealii", "sfe", "sfepy"],
    "流体动力学": ["fluids", "cantera", "openfoam", "fluidfft"],
    "分子模拟": ["ase", "pymatgen", "phonopy", "wannier90", "mdanalysis", "rdkit", "openmm", "mdtraj"],
    "天体物理": ["astropy", "gadget", "yt", "halotools", "21cmfast"],
    "地球科学": ["obspy", "pyrocko", "climlab", "iris", "xarray"],
    "生物信息": ["biopython", "pysam", "htseq", "scanpy"],
    "并行计算": ["cupy", "numba", "pycuda", "pyopencl", "dask", "ray", "joblib"],
    "数据分析": ["seaborn", "plotly", "bokeh", "geopandas", "vaex", "holoviews", "datashader"],
    "图像处理": ["opencv", "scikit-image", "pillow", "imageio", "albumentations", "mahotas"],
    "信号处理": ["librosa", "pykalman", "pywavelets"],
    "优化算法": ["nlopt", "cvxpy", "pulp", "pyomo", "openmdao", "pydoe"],
    "统计分析": ["statsmodels", "pingouin", "arviz", "pymc", "emcee", "lmfit", "chaospy", "bambi"],
    "网络分析": ["networkx", "igraph", "graph-tool", "snap", "stellargraph"],
    "地理信息": ["geopandas", "rasterio", "fiona", "shapely", "pyproj"],
    "时间序列": ["prophet", "arch", "pyflux", "tslearn"],
    "数据存储": ["zarr", "h5py", "tables", "netcdf4"],
    "其他工具": ["cython", "pytensor", "sagemath", "pythran", "pyfftw"]
}

for category, libs in categories.items():
    count = len([lib for lib in libs if os.path.exists(os.path.join(target_dir, lib))])
    print(f"{category}: {count}/{len(libs)}")

print("\n这些仓库具有以下特点：")
print("1. 复杂的跨文件依赖关系")
print("2. 专业的科学计算应用场景")
print("3. 丰富的测试用例和文档")
print("4. 活跃的开发和维护")
print("5. 良好的环境配置")
print("6. 适合构建推理代码执行benchmark")
print("7. 涵盖多个科学计算领域")
