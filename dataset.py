import os
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import opensmile
import pandas as pd

class AudioDataset(Dataset):
    def __init__(self, root_dir, info_path, event_classes=None, age_threshold=None,
                 age_option=None, gender=None, duration=1, fold_index=0, mode='train'):
        self.duration = duration
        self.fold_index = fold_index
        self.mode = mode
        info_data = np.load(info_path)
        self.subject_info = {sid: {'age': age, 'gender': gender, 'viq': viq}
                             for sid, age, gender, viq in zip(info_data['ids'], info_data['ages'],
                                                              info_data['genders'], info_data['viqs'])}
        event_map = {
            0: ['youzi'],
            1: ['wuzi', 'new_wordless'],
            2: ['shijian', 'past'],
            3: ['jiaqi', 'new_future']
        }

        files = []
        for label, category in enumerate(['TD', 'ASD']):
            category_dir = os.path.join(root_dir, category)
            for file_name in os.listdir(category_dir):
                if file_name.endswith('.wav'):
                    prefix, subject_id = file_name[:-4].rsplit('_', 1)
                    subject_id = int(subject_id)

                    event = next((k for k, v in event_map.items() if prefix in v), None)

                    if self._check_conditions(subject_id, event, age_threshold, age_option, gender, event_classes):
                        files.append((os.path.join(category_dir, file_name), label, subject_id, event))


        subjects = list(set(f[2] for f in files))
        subject_labels = [next(f[1] for f in files if f[2] == s) for s in subjects]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = list(skf.split(subjects, subject_labels))[fold_index]
        split_subjects = set(subjects[i] for i in (train_idx if mode == 'train' else test_idx))
        self.file_list = [f for f in files if f[2] in split_subjects]

        self.samples = []
        print('Select files:', self.file_list)
        for file_path, label, subject_id, event in tqdm(self.file_list, desc='loading data:'):
            waveform, sr = torchaudio.load(file_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            seg_length = int(16000 * duration)
            num_segments = waveform.shape[1] // seg_length
            for i in range(num_segments):
                seg = waveform[:, i * seg_length:(i + 1) * seg_length]
                info = self.subject_info[subject_id]
                self.samples.append((seg, label, info['age'], info['gender'], info['viq'], event))

    def load_excel_features(self, file_path, i, feature_root, duration):
        basename = os.path.basename(file_path)  # e.g., jiaqi_1.wav
        prefix = os.path.splitext(basename)[0]  # -> jiaqi_1
        label_dir = 'ASD' if 'ASD' in file_path else 'TD'
        excel_file = os.path.join(
            feature_root,
            f"{duration}s",
            label_dir,
            f"{prefix}_index_{i}.xlsx"
        )
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Feature file not found: {excel_file}")
        df = pd.read_excel(excel_file)
        feat_np = df.values.squeeze()  # shape: [N]
        return feat_np.astype(np.float32)

    def _check_conditions(self, subject_id, event, age_threshold, age_option, gender, event_classes):
        info = self.subject_info.get(subject_id)
        if info is None:
            return False
        if event_classes is not None and event not in event_classes:
            return False
        if age_threshold is not None:
            if age_option == 'older' and info['age'] <= age_threshold:
                return False
            elif age_option == 'younger' and info['age'] >= age_threshold:
                return False
        if gender is not None and info['gender'] != gender:
            return False
        return True

    def waveform_to_fbank(self, waveform, sr=16000, n_fft=400, hop_length=160, n_mels=80):

        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)  # [L]

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)  # [n_mels, T]
        log_mel_spec = torch.log(mel_spec + 1e-10)

        log_spec = torch.clamp(log_mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, a, g, v, e = self.samples[idx]
        return x, torch.tensor(y), torch.tensor(a), torch.tensor(g), torch.tensor(v), torch.tensor(e)


class AudioFeatureDataset(Dataset):
    def __init__(self, feature_root, root_dir, info_path, event_classes=None, age_threshold=None,
                 age_option=None, gender=None, duration=1, fold_index=0, mode='train'):
        self.duration = duration
        self.fold_index = fold_index
        self.mode = mode
        self.feature_root = feature_root

        info_data = np.load(info_path)
        self.subject_info = {sid: {'age': age, 'gender': gender, 'viq': viq}
                             for sid, age, gender, viq in zip(info_data['ids'], info_data['ages'],
                                                              info_data['genders'], info_data['viqs'])}
        self.X = []
        self.y = []

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        event_map = {
            0: ['youzi'],
            1: ['wuzi', 'new_wordless'],
            2: ['shijian', 'past'],
            3: ['jiaqi', 'new_future']
        }

        files = []
        for label, category in enumerate(['TD', 'ASD']):
            category_dir = os.path.join(root_dir, category)
            for file_name in os.listdir(category_dir):
                if file_name.endswith('.wav'):
                    prefix, subject_id = file_name[:-4].rsplit('_', 1)
                    subject_id = int(subject_id)

                    event = next((k for k, v in event_map.items() if prefix in v), None)

                    if self._check_conditions(subject_id, event, age_threshold, age_option, gender, event_classes):
                        files.append((os.path.join(category_dir, file_name), label, subject_id, event))


        subjects = list(set(f[2] for f in files))
        subject_labels = [next(f[1] for f in files if f[2] == s) for s in subjects]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, test_idx = list(skf.split(subjects, subject_labels))[fold_index]
        split_subjects = set(subjects[i] for i in (train_idx if mode == 'train' else test_idx))
        self.file_list = [f for f in files if f[2] in split_subjects]

        self.samples = []
        print('Select files:', self.file_list)
        for file_path, label, subject_id, event in tqdm(self.file_list, desc='loading data:'):
            waveform, sr = torchaudio.load(file_path)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            seg_length = int(16000 * duration)
            num_segments = waveform.shape[1] // seg_length
            for i in range(num_segments):
                ##------------------------------------------------------------------------------------------------------
                try:
                    feat_np = self.load_excel_features(file_path, i,
                                                  feature_root='',
                                                  duration=duration)
                except Exception as e:
                    print(f"[ERROR] Cannot load features for {file_path} index {i}: {e}")
                    continue
                self.X.append(feat_np)
                self.y.append(label)
                ##------------------------------------------------------------------------------------------------------
                seg = waveform[:, i * seg_length:(i + 1) * seg_length]
                info = self.subject_info[subject_id]
                self.samples.append((seg, label, info['age'], info['gender'], info['viq'], event))

    def load_excel_features(self, file_path, i, feature_root, duration):
        basename = os.path.basename(file_path)  # e.g., jiaqi_1.wav
        prefix = os.path.splitext(basename)[0]  # -> jiaqi_1
        label_dir = 'ASD' if 'ASD' in file_path else 'TD'
        excel_file = os.path.join(
            feature_root,
            f"{duration}s",
            label_dir,
            f"{prefix}_index_{i}.xlsx"
        )
        if not os.path.exists(excel_file):
            raise FileNotFoundError(f"Feature file not found: {excel_file}")
        df = pd.read_excel(excel_file)
        feat_np = df.values.squeeze()  # shape: [N]
        return feat_np.astype(np.float32)

    def features(self):
        return np.stack(self.X)

    def labels(self):
        return np.array(self.y)

    def _check_conditions(self, subject_id, event, age_threshold, age_option, gender, event_classes):
        info = self.subject_info.get(subject_id)
        if info is None:
            return False
        if event_classes is not None and event not in event_classes:
            return False
        if age_threshold is not None:
            if age_option == 'older' and info['age'] <= age_threshold:
                return False
            elif age_option == 'younger' and info['age'] >= age_threshold:
                return False
        if gender is not None and info['gender'] != gender:
            return False
        return True

    def waveform_to_fbank(self, waveform, sr=16000, n_fft=400, hop_length=160, n_mels=80):
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)  # [L]

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        mel_spec = mel_transform(waveform)  # [n_mels, T]
        log_mel_spec = torch.log(mel_spec + 1e-10)

        log_spec = torch.clamp(log_mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, a, g, v, e = self.samples[idx]
        #feature = torch.FloatTensor(self.X[idx][[78, 81, 15, 33, 62]])
        feature = torch.FloatTensor(self.X[idx])
        #fbank = self.waveform_to_fbank(x)  # [80, T]
        return x, torch.tensor(y), torch.tensor(a), torch.tensor(g), torch.tensor(v), torch.tensor(e), feature

if __name__ == "__main__":
    test_dataset = AudioFeatureDataset(
        root_dir='',
        info_path='',
        feature_root='',
        event_classes=[0, 1, 2, 3],
        age_threshold=None,
        age_option=None,
        gender=None,
        duration=10,
        fold_index=1,
        mode='test')
    print(test_dataset.X[0].shape)
    x, y, a, g, v, e, f = test_dataset[0]
    print(f.shape)
