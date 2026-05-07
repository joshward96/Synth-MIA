import json
import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
_TABSYN_DIR = os.path.join(_MODELS_DIR, 'TabSyn')
_RTF_SRC_DIR = os.path.join(_MODELS_DIR, 'REaLTabFormer', 'src')


class _SynthResult:
    def __init__(self, df):
        self._df = df

    def dataframe(self):
        return self._df


# --- TabSyn wrapper ---

def _prep_tabsyn_data(df, dataname):
    data_dir = os.path.join(_TABSYN_DIR, 'data', dataname)
    info_dir = os.path.join(_TABSYN_DIR, 'data', 'Info')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    df.to_csv(os.path.join(data_dir, f'{dataname}.csv'), index=False)

    col_names = df.columns.tolist()
    target_col = col_names[-1]
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    num_col_idx = [col_names.index(c) for c in num_cols if c != target_col]
    cat_col_idx = [col_names.index(c) for c in cat_cols if c != target_col]
    target_col_idx = [col_names.index(target_col)]

    last_col = df.iloc[:, -1]
    if pd.api.types.is_numeric_dtype(last_col):
        n_unique = last_col.nunique()
        task_type = 'regression' if n_unique > 20 else ('binclass' if n_unique == 2 else 'multiclass')
    else:
        task_type = 'binclass' if last_col.nunique() == 2 else 'multiclass'

    info = {
        "name": dataname,
        "task_type": task_type,
        "header": "infer",
        "column_names": None,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": f"./data/{dataname}/{dataname}.csv",
        "test_path": None,
    }
    with open(os.path.join(info_dir, f'{dataname}.json'), 'w') as f:
        json.dump(info, f, indent=4)


class _TabSynWrapper:
    def __init__(self, dataname):
        self.dataname = dataname

    def generate(self, n_samples):
        save_path = os.path.join(_TABSYN_DIR, 'synthetic', self.dataname, 'tabsyn.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        subprocess.run(
            [sys.executable, '-m', 'tabsyn.sample',
             '--dataname', self.dataname,
             '--save_path', save_path],
            cwd=_TABSYN_DIR, check=True,
        )
        df = pd.read_csv(save_path)
        return _SynthResult(df.head(n_samples))


def train_sample_tabsyn(dataset, random_state=1):
    dataname = f'tmp_{random_state}_{os.getpid()}'
    _prep_tabsyn_data(dataset, dataname)
    for cmd in [
        [sys.executable, 'process_dataset.py', '--dataname', dataname],
        [sys.executable, '-m', 'tabsyn.vae.main', '--dataname', dataname],
        [sys.executable, '-m', 'tabsyn.main', '--dataname', dataname],
    ]:
        subprocess.run(cmd, cwd=_TABSYN_DIR, check=True)
    return _TabSynWrapper(dataname)


# --- REaLTabFormer wrapper ---

class _RTFWrapper:
    def __init__(self, model):
        self._model = model

    def generate(self, n_samples):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        df = self._model.sample(n_samples=n_samples, device=device)
        return _SynthResult(df)


def train_sample_realtabformer(dataset, random_state=1):
    if _RTF_SRC_DIR not in sys.path:
        sys.path.insert(0, _RTF_SRC_DIR)
    from realtabformer import REaLTabFormer
    import torch

    run_dir = tempfile.mkdtemp(prefix=f'rtf_{random_state}_')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = REaLTabFormer(
        model_type='tabular',
        random_state=random_state,
        checkpoints_dir=os.path.join(run_dir, 'ckpt'),
        samples_save_dir=os.path.join(run_dir, 'samples'),
        full_save_dir=os.path.join(run_dir, 'full'),
    )
    model.fit(dataset, device=device)
    return _RTFWrapper(model)

def train_sample_bn(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("bayesian_network", random_state=random_state)
    syn_model.fit(loader)
    return syn_model
    
def train_sample_privbays(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("privbayes", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_aim(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("aim", random_state=random_state)
    syn_model.fit(loader)
    return syn_model
    
def train_sample_ddpm(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("ddpm", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_great(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("great", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_arf(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("arf", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_tvae(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("tvae", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_ctgan(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("ctgan", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_nflows(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("nflow", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_adsgan(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("adsgan", random_state=random_state)
    syn_model.fit(loader)
    return syn_model

def train_sample_pategan(dataset, random_state=1):
    loader = GenericDataLoader(dataset)
    syn_model = Plugins().get("pategan", random_state=random_state, epsilon = 1)
    syn_model.fit(loader)
    return syn_model