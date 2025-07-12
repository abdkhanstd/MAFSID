#!/usr/bin/env python3
"""
Multiclass Dataset Manager for Few-Shot Learning
Supports NSL-KDD and CICIDS2017 with multiple attack classes
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from collections import Counter
 
class MulticlassDataManager:
    """Dataset manager for multiclass few-shot learning"""
    
    def __init__(self, dataset_name='NSL-KDD', data_dir='datasets'):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.class_names = []
        self.num_classes = 0
        
    def load_nsl_kdd(self):
        """Load NSL-KDD dataset with multiclass labels"""
        print("📂 Loading NSL-KDD dataset...")
        
        # Load training and test files
        train_file = os.path.join(self.data_dir, 'KDDTrain+.txt')
        test_file = os.path.join(self.data_dir, 'KDDTest+.txt')
        
        # Column names for NSL-KDD
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]
        
        # Load data
        train_df = pd.read_csv(train_file, names=columns)
        test_df = pd.read_csv(test_file, names=columns)
        
        # Combine for preprocessing
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Preprocess features
        df = self._preprocess_nsl_kdd_features(df)
        
        # Get multiclass labels
        attack_mapping = {
            'normal': 'normal',
            # DoS attacks
            'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos', 'smurf': 'dos',
            'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos',
            'udpstorm': 'dos',
            # Probe attacks
            'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
            'mscan': 'probe', 'saint': 'probe',
            # R2L attacks
            'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
            'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l',
            'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpread': 'r2l', 'snmpgetattack': 'r2l',
            'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
            # U2R attacks
            'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
            'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
        }
        
        df['attack_category'] = df['label'].map(attack_mapping)
        df = df.dropna(subset=['attack_category'])
        
        return self._prepare_dataset(df, 'attack_category')
    
    def load_cicids2017(self):
        """Load CICIDS2017 dataset with multiclass labels"""
        print("📂 Loading CICIDS2017 dataset...")
        
        # Load all CSV files
        csv_files = [
            'Monday-WorkingHours.pcap_ISCX.csv',
            'Tuesday-WorkingHours.pcap_ISCX.csv',
            'Wednesday-workingHours.pcap_ISCX.csv',
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
        ]
        
        dataframes = []
        for file in csv_files:
            file_path = os.path.join(self.data_dir, 'CICIDS2017/MachineLearningCVE', file)
            if os.path.exists(file_path):
                print(f"   Loading {file}...")
                try:
                    df = pd.read_csv(file_path)
                    # Clean column names
                    df.columns = df.columns.str.strip()
                    dataframes.append(df)
                except Exception as e:
                    print(f"   ⚠️ Error loading {file}: {e}")
        
        if not dataframes:
            raise ValueError("No CICIDS2017 files found!")
        
        # Combine all dataframes
        df = pd.concat(dataframes, ignore_index=True)
        
        # Preprocess features
        df = self._preprocess_cicids2017_features(df)
        
        # The label column might be named differently
        label_column = None
        for col in ['Label', 'label', ' Label']:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("Label column not found in CICIDS2017 data")
        
        # Clean labels and create multiclass categories
        df[label_column] = df[label_column].str.strip()
        
        # Group similar attacks
        attack_mapping = {
            'BENIGN': 'benign',
            'Bot': 'botnet',
            'PortScan': 'portscan',
            'DDoS': 'ddos',
            'DoS Hulk': 'dos',
            'DoS GoldenEye': 'dos',
            'DoS slowloris': 'dos',
            'DoS Slowhttptest': 'dos',
            'FTP-Patator': 'brute_force',
            'SSH-Patator': 'brute_force',
            'Web Attack – Brute Force': 'web_attack',
            'Web Attack – XSS': 'web_attack',
            'Web Attack – Sql Injection': 'web_attack',
            'Infiltration': 'infiltration',
            'Heartbleed': 'heartbleed'
        }
        
        df['attack_category'] = df[label_column].map(attack_mapping)
        df = df.dropna(subset=['attack_category'])
        
        return self._prepare_dataset(df, 'attack_category')
    
    def load_bot_iot(self):
        """Load Bot-IoT dataset with multiclass labels"""
        print("📂 Loading Bot-IoT dataset...")
        
        # Load CSV files from Bot-IoT directory
        bot_iot_dir = os.path.join(self.data_dir, 'Bot-IoT')
        if not os.path.exists(bot_iot_dir):
            raise ValueError(f"Bot-IoT directory not found: {bot_iot_dir}")
        
        csv_files = [f for f in os.listdir(bot_iot_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(f"No CSV files found in {bot_iot_dir}")
        
        dataframes = []
        for file in csv_files:
            file_path = os.path.join(bot_iot_dir, file)
            print(f"   Loading {file}...")
            try:
                df = pd.read_csv(file_path)
                # Clean column names
                df.columns = df.columns.str.strip()
                dataframes.append(df)
            except Exception as e:
                print(f"   ⚠️ Error loading {file}: {e}")
        
        if not dataframes:
            raise ValueError("No Bot-IoT files could be loaded!")
        
        # Combine all dataframes
        df = pd.concat(dataframes, ignore_index=True)
        
        # Preprocess features
        df = self._preprocess_bot_iot_features(df)
        
        # Find label column
        label_column = None
        for col in ['category', 'attack', 'label', 'Label', 'subcategory']:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("Label column not found in Bot-IoT data")
        
        # Clean labels and create multiclass categories
        df[label_column] = df[label_column].astype(str).str.strip()
        
        # Map Bot-IoT attacks to categories
        attack_mapping = {
            'Normal': 'normal',
            'DDoS': 'ddos',
            'DoS': 'dos', 
            'Reconnaissance': 'reconnaissance',
            'Theft': 'theft',
            'Data_Exfiltration': 'data_exfiltration',
            # Additional mappings for other possible labels
            'UDP': 'dos',
            'TCP': 'dos',
            'HTTP': 'dos',
            'Service_Scanning': 'reconnaissance',
            'OS_Fingerprinting': 'reconnaissance',
            'Keylogging': 'theft',
            'Data_Theft': 'theft',
        }
        
        # Apply mapping
        df['attack_category'] = df[label_column].map(attack_mapping)
        
        # Handle unmapped categories (convert to lowercase and replace spaces)
        mask = df['attack_category'].isna()
        if mask.any():
            df.loc[mask, 'attack_category'] = df.loc[mask, label_column].str.lower().str.replace(' ', '_')
        
        df = df.dropna(subset=['attack_category'])
        
        return self._prepare_dataset(df, 'attack_category')
    
    def load_in_vehicle(self):
        """Load In-Vehicle Network Intrusion Detection dataset"""
        print("📂 Loading In-Vehicle Network dataset...")
        
        # Define the base path
        base_path = os.path.join(self.data_dir, 'In-Vehicle Network Intrusion Detection')
        
        # Get all training folders
        train_folders = [
            'car_track_final_1st_train',
            'car_track_final_2nd_train', 
            'car_track_preliminary_train'
        ]
        
        dataframes = []
        
        for folder in train_folders:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                print(f"   Loading from {folder}...")
                
                # Get all CSV files in this folder
                csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    file_path = os.path.join(folder_path, csv_file)
                    print(f"     Processing {csv_file}...")
                    
                    try:
                        # Extract attack type from filename
                        filename_lower = csv_file.lower()
                        
                        if 'attack_free' in filename_lower or 'normal' in filename_lower:
                            attack_type = 'Attack_free'  # Use original case
                        elif 'fuzzy' in filename_lower:
                            attack_type = 'Fuzzy'
                        elif 'malfunction' in filename_lower:
                            attack_type = 'Malfunction'
                        elif 'flooding' in filename_lower:
                            attack_type = 'Flooding'
                        elif 'replay' in filename_lower:
                            attack_type = 'Replay'
                        elif 'dos' in filename_lower:
                            attack_type = 'DoS'
                        elif 'file_' in filename_lower:
                            # Handle generic File_X.csv names - assume normal if no attack keyword
                            attack_type = 'Normal'
                        else:
                            # Extract first word before underscore as attack type
                            parts = csv_file.split('_')
                            if len(parts) > 0:
                                attack_type = parts[0]  # Keep original case
                            else:
                                attack_type = 'Unknown'
                        
                        # Read CSV file
                        df = pd.read_csv(file_path, header=None)
                        
                        # Assign column names based on actual structure (4 columns: timestamp, can_id, dlc, data)
                        if df.shape[1] == 4:
                            df.columns = ['timestamp', 'can_id', 'dlc', 'data']
                        else:
                            # Fallback for unexpected structures
                            df.columns = [f'col_{i}' for i in range(df.shape[1])]
                            if df.shape[1] >= 4:
                                df = df.rename(columns={'col_0': 'timestamp', 'col_1': 'can_id', 'col_2': 'dlc', 'col_3': 'data'})
                        
                        # Add attack type label
                        df['attack_category'] = attack_type
                        
                        # Extract vehicle type from filename if present
                        if 'sonata' in filename_lower:
                            df['vehicle'] = 'hyundai_sonata'
                        elif 'soul' in filename_lower:
                            df['vehicle'] = 'kia_soul'
                        elif 'spark' in filename_lower:
                            df['vehicle'] = 'chevrolet_spark'
                        else:
                            df['vehicle'] = 'unknown'
                        
                        dataframes.append(df)
                        
                    except Exception as e:
                        print(f"     ⚠️ Error loading {csv_file}: {e}")
        
        if not dataframes:
            raise ValueError("No In-Vehicle Network files found!")
        
        # Combine all dataframes
        df = pd.concat(dataframes, ignore_index=True)
        
        # Preprocess features
        df = self._preprocess_in_vehicle_features(df)
        
        return self._prepare_dataset(df, 'attack_category')
        
        # Preprocess features
        df = self._preprocess_in_vehicle_features(df)
        
        return self._prepare_dataset(df, 'attack_category')
    
    def load_car_hacking(self):
        """Load Car Hacking dataset"""
        print("📂 Loading Car Hacking dataset...")
        
        car_hacking_path = os.path.join(self.data_dir, 'carhacking')
        
        dataframes = []
        
        # File to attack type mapping based on filename
        file_attack_mapping = {
            'normal_run_data.txt': 'Normal',
            'DoS_dataset.csv': 'DoS',
            'Fuzzy_dataset.csv': 'Fuzzy', 
            'gear_dataset.csv': 'Gear',
            'RPM_dataset.csv': 'RPM'
        }
        
        for filename, attack_type in file_attack_mapping.items():
            file_path = os.path.join(car_hacking_path, filename)
            
            if os.path.exists(file_path):
                print(f"   Loading {filename} ({attack_type})...")
                
                try:
                    if filename.endswith('.txt'):
                        # Handle the normal data file with different format
                        df = self._parse_car_hacking_txt(file_path)
                    else:
                        # Handle CSV files with variable field counts
                        df = self._parse_car_hacking_csv(file_path)
                        
                    df['attack_category'] = attack_type
                    dataframes.append(df)
                    
                except Exception as e:
                    print(f"   ⚠️ Error loading {filename}: {e}")
        
        if not dataframes:
            raise ValueError("No Car Hacking files found!")
        
        # Combine all dataframes
        df = pd.concat(dataframes, ignore_index=True)
        
        # Preprocess features
        df = self._preprocess_car_hacking_features(df)
        
        return self._prepare_dataset(df, 'attack_category')
    
    def _parse_car_hacking_txt(self, file_path):
        """Parse the normal_run_data.txt file with custom format"""
        data = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse line format: "Timestamp: X.Y        ID: ZZZZ    000    DLC: N    data bytes"
                    parts = line.split()
                    
                    # Extract timestamp
                    timestamp_idx = parts.index('Timestamp:')
                    timestamp = float(parts[timestamp_idx + 1])
                    
                    # Extract CAN ID 
                    id_idx = parts.index('ID:')
                    can_id = parts[id_idx + 1]
                    can_id_int = int(can_id, 16)
                    
                    # Extract DLC
                    dlc_idx = parts.index('DLC:')
                    dlc = int(parts[dlc_idx + 1])
                    
                    # Extract data bytes (after DLC value)
                    data_bytes = parts[dlc_idx + 2:]
                    
                    # Convert hex bytes to integers and pad to 8 bytes
                    data_values = []
                    for byte_str in data_bytes[:8]:  # Take up to 8 bytes
                        try:
                            data_values.append(int(byte_str, 16))
                        except ValueError:
                            data_values.append(0)
                    
                    # Pad to 8 data bytes
                    while len(data_values) < 8:
                        data_values.append(0)
                    
                    # Create row: [timestamp, can_id, dlc, data_0, data_1, ..., data_7]
                    row = [timestamp, can_id_int, dlc] + data_values
                    data.append(row)
                    
                except (ValueError, IndexError) as e:
                    # Skip malformed lines
                    continue
        
        # Create DataFrame with consistent column names
        columns = ['timestamp', 'can_id', 'dlc'] + [f'data_{i}' for i in range(8)]
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def _parse_car_hacking_csv(self, file_path):
        """Parse Car Hacking CSV files with variable field counts"""
        data = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Split by comma
                    parts = line.split(',')
                    
                    if len(parts) >= 4:  # minimum: timestamp, can_id, dlc, flag
                        timestamp = float(parts[0])
                        can_id = parts[1]
                        dlc = int(parts[2])
                        flag = parts[-1]  # last column is always flag
                        
                        # Data bytes are between dlc and flag
                        data_bytes = parts[3:-1]
                        
                        # Convert hex bytes to integers and pad/truncate to 8 bytes
                        data_values = []
                        for byte_str in data_bytes[:8]:  # Take up to 8 bytes
                            try:
                                data_values.append(int(byte_str, 16))
                            except ValueError:
                                data_values.append(0)
                        
                        # Pad to 8 data bytes if needed
                        while len(data_values) < 8:
                            data_values.append(0)
                        
                        # Create standardized row: [timestamp, can_id, dlc, data_0..data_7, flag]
                        can_id_int = int(can_id, 16) if can_id else 0
                        row = [timestamp, can_id_int, dlc] + data_values + [flag]
                        data.append(row)
                        
                except (ValueError, IndexError) as e:
                    # Skip malformed lines but continue processing
                    if line_num % 100000 == 0:  # Log every 100k lines for progress
                        print(f"     Skipping malformed line {line_num}")
                    continue
        
        # Create DataFrame with consistent columns
        columns = ['timestamp', 'can_id', 'dlc'] + [f'data_{i}' for i in range(8)] + ['flag']
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def _preprocess_nsl_kdd_features(self, df):
        """Preprocess NSL-KDD features"""
        # Remove difficulty column
        df = df.drop(['difficulty'], axis=1)
        
        # Handle categorical features
        categorical_features = ['protocol_type', 'service', 'flag']
        df = pd.get_dummies(df, columns=categorical_features)
        
        return df
    
    def _preprocess_cicids2017_features(self, df):
        """Preprocess CICIDS2017 features"""
        # Remove non-numeric columns that aren't labels
        label_cols = ['Label', 'label', ' Label']
        feature_cols = []
        
        for col in df.columns:
            if col not in label_cols:
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    feature_cols.append(col)
                except:
                    pass
        
        # Keep only numeric features and labels
        keep_cols = feature_cols + [col for col in label_cols if col in df.columns]
        df = df[keep_cols]
        
        # Handle missing values
        df = df.fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def _preprocess_bot_iot_features(self, df):
        """Preprocess Bot-IoT features"""
        # Remove non-numeric columns that aren't useful for ML
        exclude_cols = ['saddr', 'daddr', 'smac', 'dmac', 'soui', 'doui', 'flgs', 'proto', 'state']
        
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and not any(keyword in col.lower() for keyword in ['label', 'attack', 'category']):
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    feature_cols.append(col)
                except:
                    # Handle categorical columns by encoding them
                    if df[col].dtype == 'object' and col not in exclude_cols:
                        # Simple label encoding for categorical features
                        unique_vals = df[col].unique()
                        if len(unique_vals) < 50:  # Only encode if not too many categories
                            df[col] = pd.Categorical(df[col]).codes
                            feature_cols.append(col)
        
        # Keep only feature columns and labels
        label_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['label', 'attack', 'category'])]
        keep_cols = feature_cols + label_cols
        df = df[keep_cols]
        
        # Handle missing values
        df = df.fillna(0)
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def _preprocess_in_vehicle_features(self, df):
        """Preprocess In-Vehicle Network features"""
        # Convert CAN ID from hex string to integer
        if 'can_id' in df.columns and df['can_id'].dtype == 'object':
            df['can_id'] = df['can_id'].apply(
                lambda x: int(str(x), 16) if pd.notna(x) and str(x).strip() != '' else 0
            )
        
        # Parse data field (hex bytes separated by spaces)
        if 'data' in df.columns:
            # Handle space-separated hex bytes
            data_split = df['data'].str.split(' ', expand=True)
            
            # Convert hex bytes to integers (up to 8 bytes for CAN)
            max_bytes = min(8, data_split.shape[1] if data_split is not None and data_split.shape[1] else 0)
            for i in range(max_bytes):
                col_name = f'data_{i}'
                if i < data_split.shape[1]:
                    def safe_hex_convert(x):
                        try:
                            if pd.notna(x) and str(x).strip() != '':
                                return int(str(x), 16)
                            else:
                                return 0
                        except ValueError:
                            return 0
                    
                    df[col_name] = data_split[i].apply(safe_hex_convert)
                else:
                    df[col_name] = 0
            
            # Add remaining data bytes as 0 if less than 8
            for i in range(max_bytes, 8):
                df[f'data_{i}'] = 0
            
            # Drop original data column
            df = df.drop(['data'], axis=1)
        
        # Convert categorical features using one-hot encoding
        if 'vehicle' in df.columns:
            df = pd.get_dummies(df, columns=['vehicle'], prefix='vehicle')
        
        # Handle any remaining non-numeric columns
        for col in df.columns:
            if col not in ['attack_category'] and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values and handle infinite values
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def _preprocess_car_hacking_features(self, df):
        """Preprocess Car Hacking features"""
        # Convert CAN ID to numeric if it's hex string
        if 'can_id' in df.columns:
            if df['can_id'].dtype == 'object':
                # Handle hex strings (remove 0x prefix if present)
                df['can_id'] = df['can_id'].apply(
                    lambda x: int(str(x).replace('0x', ''), 16) if pd.notna(x) and str(x).strip() != '' else 0
                )
            df['can_id'] = pd.to_numeric(df['can_id'], errors='coerce')
        
        # Handle hex data columns if they're strings
        for col in df.columns:
            if col.startswith('data_'):
                if df[col].dtype == 'object':
                    # Convert hex strings to integers, handle non-hex values
                    def safe_hex_convert(x):
                        try:
                            if pd.notna(x) and str(x).strip() != '':
                                return int(str(x), 16)
                            else:
                                return 0
                        except ValueError:
                            return 0
                    
                    df[col] = df[col].apply(safe_hex_convert)
        
        # Handle flag column if present (convert to categorical)
        if 'flag' in df.columns:
            df = pd.get_dummies(df, columns=['flag'], prefix='flag')
        
        # Ensure all remaining columns are numeric
        for col in df.columns:
            if col not in ['attack_category']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def _prepare_dataset(self, df, label_column):
        """Prepare dataset for few-shot learning"""
        # Separate features and labels
        features = df.drop([label_column], axis=1)
        labels = df[label_column]
        
        # Remove any remaining non-numeric columns
        numeric_features = []
        for col in features.columns:
            try:
                features[col] = pd.to_numeric(features[col], errors='coerce')
                numeric_features.append(col)
            except:
                pass
        
        features = features[numeric_features]
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_
        self.num_classes = len(self.class_names)
        
        print(f"📊 Dataset Info:")
        print(f"   Classes: {self.class_names}")
        print(f"   Number of classes: {self.num_classes}")
        print(f"   Features: {features.shape[1]}")
        print(f"   Total samples: {len(features)}")
        
        # Print class distribution
        class_counts = Counter(labels)
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count}")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features_scaled, labels_encoded, test_size=0.4, random_state=42, stratify=labels_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'train': {'features': X_train, 'labels': y_train},
            'val': {'features': X_val, 'labels': y_val},
            'test': {'features': X_test, 'labels': y_test},
            'feature_dim': features_scaled.shape[1],
            'num_classes': self.num_classes,
            'class_names': self.class_names
        }
    
    def load_dataset(self):
        """Load the specified dataset"""
        if self.dataset_name == 'NSL-KDD':
            return self.load_nsl_kdd()
        elif self.dataset_name == 'CICIDS2017':
            return self.load_cicids2017()
        elif self.dataset_name == 'In-Vehicle':
            return self.load_in_vehicle()
        elif self.dataset_name == 'Car-Hacking':
            return self.load_car_hacking()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def save_metadata(self, filepath):
        """Save dataset metadata"""
        metadata = {
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }
        with open(filepath, 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_metadata(self, filepath):
        """Load dataset metadata"""
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)
        
        self.dataset_name = metadata['dataset_name']
        self.num_classes = metadata['num_classes']
        self.class_names = metadata['class_names']
        self.label_encoder = metadata['label_encoder']
        self.scaler = metadata['scaler']


def create_few_shot_episodes_multiclass(data, n_way=5, k_shot=5, q_query=15, num_episodes=1000):
    """Create few-shot learning episodes for multiclass classification"""
    print(f"🎯 Creating Few-Shot Episodes: {n_way}-way {k_shot}-shot")
    
    # Handle both data dictionary and dataset object
    if isinstance(data, dict) and 'features' in data:
        features = data['features']
        labels = data['labels']
        num_classes = len(np.unique(labels))
    else:
        # Assume it's a full dataset object
        features = data['train']['features']
        labels = data['train']['labels']
        num_classes = data['num_classes']
    
    if n_way > num_classes:
        print(f"⚠️ Warning: n_way ({n_way}) > num_classes ({num_classes}), using n_way={num_classes}")
        n_way = num_classes
    
    episodes = []
    
    for episode_idx in range(num_episodes):
        # Randomly sample n_way classes
        selected_classes = np.random.choice(num_classes, n_way, replace=False)
        
        episode_support_x = []
        episode_support_y = []
        episode_query_x = []
        episode_query_y = []
        
        for way_idx, class_id in enumerate(selected_classes):
            # Get all samples for this class
            class_mask = labels == class_id
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) < k_shot + q_query:
                # If not enough samples, sample with replacement
                selected_indices = np.random.choice(class_indices, k_shot + q_query, replace=True)
            else:
                selected_indices = np.random.choice(class_indices, k_shot + q_query, replace=False)
            
            # Split into support and query
            support_indices = selected_indices[:k_shot]
            query_indices = selected_indices[k_shot:k_shot + q_query]
            
            # Add to episode
            episode_support_x.append(features[support_indices])
            episode_support_y.extend([way_idx] * k_shot)  # Use way_idx for episode-specific labeling
            
            episode_query_x.append(features[query_indices])
            episode_query_y.extend([way_idx] * len(query_indices))
        
        # Combine support and query data
        episode_support_x = np.vstack(episode_support_x)
        episode_support_y = np.array(episode_support_y)
        episode_query_x = np.vstack(episode_query_x)
        episode_query_y = np.array(episode_query_y)
        
        # Create episode dictionary
        episode = {
            'support_x': episode_support_x,
            'support_y': episode_support_y,
            'query_x': episode_query_x,
            'query_y': episode_query_y,
            'n_way': n_way,
            'k_shot': k_shot,
            'q_query': len(query_indices)
        }
        
        episodes.append(episode)
    
    print(f"✅ Created {len(episodes)} episodes")
    return episodes
