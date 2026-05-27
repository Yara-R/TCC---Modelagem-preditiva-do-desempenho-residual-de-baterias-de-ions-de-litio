import scipy.io
import pandas as pd
import numpy as np

class MatLoader:
    def __init__(self, config=None):
        self.config = config

    def _extract_from_mat(self, path):
        """
        Carrega arquivos estruturados .mat (comum em datasets como os da NASA)
        e os converte para um DataFrame legível contendo o histórico de envelhecimento.
        """
        try:
            mat = scipy.io.loadmat(path)
            
            # Localiza a chave primária de dados (ex: 'B0005' no caso da NASA)
            key = [k for k in mat.keys() if not k.startswith('__')][0]
            struct = mat[key]
            
            # Verifica se possui a árvore de dados estruturada por ciclos
            if 'cycle' in struct.dtype.names:
                cycles = struct[0, 0]['cycle'][0]
                
                capacities = []
                voltages = []
                currents = []
                times = []
                
                for c in cycles:
                    # Filtra ciclos de descarga para acompanhar a perda de capacidade
                    if c['type'][0] == 'discharge':
                        data = c['data'][0, 0]
                        
                        # Extrai a capacidade medida do ciclo
                        cap_val = data['Capacity'][0, 0] if 'Capacity' in data.dtype.names else np.nan
                        capacities.append(float(cap_val))
                        
                        # Captura dados agregados de tensão, corrente e tempo para o ciclo
                        v_meas = data['Voltage_measured'][0] if 'Voltage_measured' in data.dtype.names else [4.2]
                        i_meas = data['Current_measured'][0] if 'Current_measured' in data.dtype.names else [1.0]
                        t_meas = data['Time'][0] if 'Time' in data.dtype.names else [0.0]
                        
                        voltages.append(float(np.max(v_meas)))
                        currents.append(float(np.mean(np.abs(i_meas))))
                        times.append(float(np.max(t_meas)))
                
                # Retorna o DataFrame sumarizado por ciclo para o treinamento do cérebro de IA
                df = pd.DataFrame({
                    'voltage': voltages,
                    'capacity': capacities,
                    'current': currents,
                    'time': times
                })
                return df
                
            else:
                # Estrutura de contingência simples para arquivos .mat planos
                data_dict = {}
                for k in mat.keys():
                    if not k.startswith('__'):
                        arr = np.array(mat[k]).flatten()
                        if len(arr) > 0:
                            data_dict[k] = arr
                return pd.DataFrame.from_dict(data_dict, orient='index').transpose()
                
        except Exception as e:
            print(f"❌ Erro ao processar o arquivo binário .mat {path}: {e}")
            return pd.DataFrame()