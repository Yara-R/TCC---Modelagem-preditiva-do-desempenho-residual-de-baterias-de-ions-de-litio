"""
mat_loader.py — Oxford + NASA + genéricos
Corrige: "setting an array element with a sequence"
         via squeeze_me=True, struct_as_record=False no scipy.
"""

import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# DIAGNÓSTICO — use quando não souber a estrutura real do arquivo
# ─────────────────────────────────────────────────────────────────

def diagnose(path: str, max_depth: int = 5) -> None:
    """
    Imprime a árvore completa de um .mat (v5 ou v7.3/HDF5).

    Uso:
        from mat_loader import diagnose
        diagnose("./data/Oxford/Oxford_Battery_Degradation_Dataset_1.mat")
        diagnose("./data/Oxford/ExampleDC_C1.mat")
    """
    path = str(path)
    print(f"\n{'='*60}\nDIAGNÓSTICO: {Path(path).name}\n{'='*60}")

    # --- tenta scipy (v5) com as opções corretas ---
    try:
        mat = scipy.io.loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False,
            mat_dtype=True,
        )
        print("[Formato] MATLAB v5 (scipy)\n")
        keys = [k for k in mat.keys() if not k.startswith("__")]
        for k in keys:
            _diag_scipy(mat[k], name=k, depth=0, max_depth=max_depth)
        return
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"[scipy erro] {e}")

    # --- tenta h5py (v7.3) ---
    try:
        import h5py
        print("[Formato] MATLAB v7.3 / HDF5 (h5py)\n")
        with h5py.File(path, "r") as f:
            _diag_hdf5(f, depth=0, max_depth=max_depth)
    except ImportError:
        print("❌ h5py não instalado:  pip install h5py")
    except Exception as e:
        print(f"[h5py erro] {e}")


def _diag_scipy(obj, name, depth, max_depth):
    indent = "  " * depth
    if depth > max_depth:
        return
    if hasattr(obj, "_fieldnames"):                          # mat_struct
        print(f"{indent}[struct] {name}  campos={obj._fieldnames}")
        for field in obj._fieldnames:
            _diag_scipy(getattr(obj, field), field, depth + 1, max_depth)
    elif isinstance(obj, np.ndarray) and obj.dtype == object:
        print(f"{indent}[cell/array-obj] {name}  shape={obj.shape}")
        for i, item in enumerate(obj.flat):
            if i >= 3:
                print(f"{indent}  ...")
                break
            _diag_scipy(item, f"{name}[{i}]", depth + 1, max_depth)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}[array] {name}  shape={obj.shape}  dtype={obj.dtype}")
    else:
        print(f"{indent}[?] {name}  type={type(obj).__name__}  val={str(obj)[:60]}")


def _diag_hdf5(node, depth, max_depth, name="(root)"):
    import h5py
    indent = "  " * depth
    if depth > max_depth:
        return
    if isinstance(node, h5py.Group):
        print(f"{indent}[group] {name}/  ({len(node)} itens)")
        for k, v in list(node.items())[:10]:
            _diag_hdf5(v, depth + 1, max_depth, k)
        if len(node) > 10:
            print(f"{indent}  ... (+{len(node)-10} mais)")
    elif isinstance(node, h5py.Dataset):
        print(f"{indent}[dataset] {name}  shape={node.shape}  dtype={node.dtype}")


# ─────────────────────────────────────────────────────────────────
# CLASSE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

class MatLoader:
    # Opções que corrigem o erro "setting an array element with a sequence"
    _SCIPY_OPTS = dict(squeeze_me=True, struct_as_record=False, mat_dtype=True)

    _FIELD_RENAME = {
        "v": "voltage", "q": "capacity", "i": "current",
        "t": "time",    "T": "temperature",
    }

    def __init__(self, config=None):
        self.config = config

    # ── ENTRADA PÚBLICA ──────────────────────────────────────────

    def _extract_from_mat(self, path: str) -> pd.DataFrame:
        path = str(path)
        try:
            mat = scipy.io.loadmat(path, **self._SCIPY_OPTS)
            return self._dispatch_scipy(mat, Path(path).name.lower())
        except NotImplementedError:
            return self._load_hdf5(path)
        except Exception as e:
            print(f"❌ Erro ao abrir {path}: {e}")
            return pd.DataFrame()

    # ── ROTEADOR scipy ───────────────────────────────────────────

    def _dispatch_scipy(self, mat: dict, filename: str) -> pd.DataFrame:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            return pd.DataFrame()

        # NASA: struct com campo 'cycle'
        for k in keys:
            obj = mat[k]
            if hasattr(obj, "_fieldnames") and "cycle" in obj._fieldnames:
                return self._parse_nasa(obj)

        # Oxford ExampleDC: struct com campos 'ch' e/ou 'dc'
        for k in keys:
            obj = mat[k]
            if hasattr(obj, "_fieldnames"):
                fl = [f.lower() for f in obj._fieldnames]
                if "dc" in fl or "ch" in fl:
                    return self._parse_oxford_exampledc(obj)

        # OXFORD DATASET 1 (MATLAB v5): Mapeia chaves como 'Cell1', 'Cell2' de topo
        oxford_cell_keys = [k for k in keys if k.lower().startswith("cell")]
        if oxford_cell_keys:
            return self._parse_oxford_v5_aggregated_cells(mat, oxford_cell_keys)

        # Fallback: tenta ler arrays de topo diretamente
        return self._parse_flat_arrays(mat, keys)

    # ── NASA ─────────────────────────────────────────────────────

    def _parse_nasa(self, struct) -> pd.DataFrame:
        cycles = np.atleast_1d(struct.cycle)
        rows = []
        for c in cycles:
            if not hasattr(c, "type") or str(c.type).strip() != "discharge":
                continue
            d = c.data
            cap = float(np.atleast_1d(d.Capacity).flat[0])  if hasattr(d, "Capacity")          else np.nan
            v   = float(np.max(np.atleast_1d(d.Voltage_measured)))  if hasattr(d, "Voltage_measured")  else np.nan
            i   = float(np.mean(np.abs(np.atleast_1d(d.Current_measured)))) if hasattr(d, "Current_measured") else np.nan
            t   = float(np.max(np.atleast_1d(d.Time)))      if hasattr(d, "Time")              else np.nan
            rows.append({"voltage": v, "capacity": cap, "current": i, "time": t})
        return pd.DataFrame(rows)

    # ── OXFORD ExampleDC_C1.mat ──────────────────────────────────

    def _parse_oxford_exampledc(self, top_struct) -> pd.DataFrame:
        target = None
        for fname in top_struct._fieldnames:
            if fname.lower() == "dc":
                target = getattr(top_struct, fname)
                break
        if target is None:
            target = getattr(top_struct, top_struct._fieldnames[0])

        return self._mat_struct_to_df(target)

    def _mat_struct_to_df(self, struct) -> pd.DataFrame:
        """Converte mat_struct com campos t/v/q/T/i em DataFrame."""
        if not hasattr(struct, "_fieldnames"):
            return pd.DataFrame()

        data = {}
        for field in struct._fieldnames:
            arr = np.atleast_1d(getattr(struct, field)).flatten()
            try:
                arr = arr.astype(float)
            except (ValueError, TypeError):
                continue
            if len(arr) > 1:
                col = self._FIELD_RENAME.get(field, field.lower())
                data[col] = arr

        return self._align(data)

    # ── OXFORD DATASET 1 (MATLAB v5 - Resiliência Baseada no HDF5) ──

    def _parse_oxford_v5_aggregated_cells(self, mat: dict, cell_keys: list) -> pd.DataFrame:
        """
        Extrai de forma estável o histórico completo de ciclos guardado dentro
        das chaves estruturadas 'Cell1'...'Cell8' do arquivo Matlab v5.
        """
        rows = []
        for cell_name in sorted(cell_keys):
            cell_struct = mat[cell_name]
            if not hasattr(cell_struct, "_fieldnames"):
                continue

            # Varre os ciclos ('cyc0100', 'cyc0200', etc.) salvos como atributos da célula
            for cyc_name in sorted(cell_struct._fieldnames):
                if not cyc_name.lower().startswith("cyc"):
                    continue
                
                cyc_struct = getattr(cell_struct, cyc_name)
                if not hasattr(cyc_struct, "_fieldnames"):
                    continue
                
                # Escolhe a subestrutura de descarga (ex: 'C1dc')
                dc_key = self._pick_dc_key_scipy(cyc_struct._fieldnames)
                if dc_key is None:
                    continue
                
                dc_struct = getattr(cyc_struct, dc_key)
                if not hasattr(dc_struct, "_fieldnames"):
                    continue

                # Extração segura e conversão numérica pura dos vetores temporais
                q_arr = np.atleast_1d(getattr(dc_struct, "q", [])).flatten()
                v_arr = np.atleast_1d(getattr(dc_struct, "v", [])).flatten()
                t_arr = np.atleast_1d(getattr(dc_struct, "t", [])).flatten()
                T_arr = np.atleast_1d(getattr(dc_struct, "T", [])).flatten()

                if len(q_arr) == 0:
                    continue

                # Extrai a capacidade máxima obtida no ciclo (último ponto da descarga)
                capacity = float(np.abs(q_arr[-1]) if q_arr[-1] != 0 else np.max(np.abs(q_arr)))
                
                rows.append({
                    "cell_id":     cell_name,
                    "cycle":       self._cycle_num(cyc_name),
                    "capacity":    capacity,
                    "voltage":     float(np.mean(v_arr)) if len(v_arr) > 0 else np.nan,
                    "time":        float(t_arr[-1])      if len(t_arr) > 0 else np.nan,
                    "temperature": float(np.mean(T_arr)) if len(T_arr) > 0 else np.nan,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(["cell_id", "cycle"]).reset_index(drop=True)
        print(f"✅ Oxford v5 Integrado: {len(df)} ciclos em {df['cell_id'].nunique()} células processadas.")
        return df

    @staticmethod
    def _pick_dc_key_scipy(fieldnames: list) -> str | None:
        """Localiza a subestrutura de descarga na lista de campos."""
        for c in ("C1dc", "c1dc", "OCVdc", "ocvdc"):
            if c in fieldnames: 
                return c
        for f in fieldnames:
            if "dc" in f.lower(): 
                return f
        return None

    # ── FALLBACK: arrays planos no topo ──────────────────────────

    def _parse_flat_arrays(self, mat: dict, keys: list) -> pd.DataFrame:
        data = {}
        for k in keys:
            arr = np.atleast_1d(mat[k]).flatten()
            try:
                arr = arr.astype(float)
            except (ValueError, TypeError):
                continue
            if len(arr) > 1:
                col = self._FIELD_RENAME.get(k, k.lower())
                data[col] = arr
        return self._align(data)

    # ── OXFORD PRINCIPAL — HDF5 (v7.3, 262 MB) ──────────────────

    def _load_hdf5(self, path: str) -> pd.DataFrame:
        try:
            import h5py
        except ImportError:
            print("❌ h5py não instalado:  pip install h5py")
            return pd.DataFrame()

        rows = []
        try:
            with h5py.File(path, "r") as f:
                cell_keys = self._hdf5_cell_keys(f)
                if not cell_keys:
                    return pd.DataFrame()

                for cell_name in sorted(cell_keys):
                    cell_grp = f[cell_name]
                    for cyc_name in sorted(cell_grp.keys()):
                        cyc_grp = cell_grp[cyc_name]
                        dc_key  = self._pick_dc_key(cyc_grp)
                        if dc_key is None:
                            continue

                        dc    = cyc_grp[dc_key]
                        q_arr = self._hdf5_read(f, dc, "q")
                        v_arr = self._hdf5_read(f, dc, "v")
                        t_arr = self._hdf5_read(f, dc, "t")
                        T_arr = self._hdf5_read(f, dc, "T")

                        if q_arr is None or len(q_arr) == 0:
                            continue

                        capacity = float(np.abs(q_arr[-1]) or np.max(np.abs(q_arr)))
                        rows.append({
                            "cell_id":     cell_name,
                            "cycle":       self._cycle_num(cyc_name),
                            "capacity":    capacity,
                            "voltage":     float(np.mean(v_arr)) if v_arr is not None else np.nan,
                            "time":        float(t_arr[-1])      if t_arr is not None else np.nan,
                            "temperature": float(np.mean(T_arr)) if T_arr is not None else np.nan,
                        })

        except Exception as e:
            print(f"❌ Erro HDF5 em {path}: {e}")
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(["cell_id", "cycle"]).reset_index(drop=True)
        print(f"✅ Oxford HDF5: {len(df)} ciclos em {df['cell_id'].nunique()} células.")
        return df

    # ── helpers HDF5 ─────────────────────────────────────────────

    @staticmethod
    def _hdf5_cell_keys(f) -> list:
        import h5py
        out = [k for k in f.keys()
               if k.lower().startswith("cell") or
               (len(k) <= 3 and k.lower().startswith("c") and k[1:].isdigit())]
        return out or [k for k in f.keys() if isinstance(f[k], h5py.Group)]

    @staticmethod
    def _pick_dc_key(grp) -> str | None:
        for c in ("C1dc", "c1dc", "OCVdc", "ocvdc"):
            if c in grp: return c
        for k in grp.keys():
            if "dc" in k.lower(): return k
        return None

    @staticmethod
    def _hdf5_read(f, grp, field: str):
        import h5py
        if field not in grp:
            return None
        try:
            raw = grp[field]
            if isinstance(raw, h5py.Dataset):
                if h5py.check_ref_dtype(raw.dtype) or raw.dtype == object:
                    return np.array([f[r][()] for r in raw.flat]).flatten().astype(float)
                return np.array(raw).flatten().astype(float)
        except Exception as e:
            print(f"  ⚠️  Campo '{field}': {e}")
        return None

    @staticmethod
    def _cycle_num(name: str) -> int:
        d = "".join(c for c in name if c.isdigit())
        return int(d) if d else -1

    # ── alinhamento de vetores ───────────────────────────────────

    @staticmethod
    def _align(data: dict) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()
        lengths = [len(v) for v in data.values()]
        ideal   = max(set(lengths), key=lengths.count)
        out = {}
        for col, vec in data.items():
            if len(vec) == ideal:
                out[col] = vec
            elif len(vec) > ideal:
                out[col] = vec[:ideal]
            else:
                out[col] = np.pad(vec, (0, ideal - len(vec)), "edge")
        return pd.DataFrame(out)