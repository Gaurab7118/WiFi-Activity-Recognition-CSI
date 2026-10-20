import os
import glob
import time
import numpy as np
import pandas as pd
import csiread
from tqdm import tqdm

ROOM_A_CLASSES = ['fall', 'sit', 'sitdown', 'stand', 'standup', 'walk']

ROOM_A_NAME_MAP = {
    'sit': 'sit',
    'stand': 'stand',
    'walk': 'walk',
    'fall': 'fall',
    'sitdown': 'sitdown',
    'standup': 'standup'
}

ROOM_B_NAME_MAP = {
    'Sitting': 'sit',
    'Standing': 'stand',
    'Walking': 'walk',
    'Empty': 'empty',
    'Lying': 'lying'
}

SHARED_ACTIVITIES = ['sit', 'stand', 'walk']
ROOM_A_EXCLUSIVE = ['fall', 'sitdown', 'standup']
ROOM_B_EXCLUSIVE = ['empty', 'lying']
ALL_ACTIVITIES = ['sit', 'stand', 'walk', 'fall', 'sitdown', 'standup', 'empty', 'lying']
ACTIVITY_TO_ID = {name: idx for idx, name in enumerate(ALL_ACTIVITIES)}

def window_csi_sequence(sequence, window_size=150):
    num_windows = len(sequence) // window_size
    if num_windows == 0:
        return np.zeros((0, window_size, 256), dtype=np.float32)
    windows = sequence[:num_windows * window_size]
    return windows.reshape(num_windows, window_size, 256).astype(np.float32)

def load_pcap_file(pcap_path):
    try:
        csidata = csiread.Nexmon(pcap_path, chip='4366c0', bw=80)
        csidata.read()
        if csidata.csi is None:
            return None
        complex_csi = csidata.csi
        if complex_csi.ndim > 2:
            complex_csi = complex_csi[:, 0, 0, :]
        amplitude = np.abs(complex_csi)
        return amplitude
    except Exception as e:
        return None

def process_room_a(base_path, is_val=False):
    windows_list = []
    labels_list = []
    capture_ids = []
    file_names = []
    capture_id = 0
    
    classes_to_process = ROOM_A_CLASSES
    print(f"Processing Room A ({'validation' if is_val else 'development'})...")
    
    for cls in tqdm(classes_to_process, desc="Room A classes"):
        if is_val:
            cls_folder = os.path.join(base_path, "validation", cls)
        else:
            cls_folder = os.path.join(base_path, cls)
            
        pcap_files = sorted(glob.glob(os.path.join(cls_folder, "*.pcap")))
        act_id = ACTIVITY_TO_ID[ROOM_A_NAME_MAP[cls]]
        
        for f in pcap_files:
            amp = load_pcap_file(f)
            if amp is None:
                continue
            wins = window_csi_sequence(amp, window_size=150)
            if len(wins) > 0:
                windows_list.append(wins)
                labels_list.extend([act_id] * len(wins))
                capture_ids.extend([capture_id] * len(wins))
                file_names.extend([os.path.basename(f)] * len(wins))
                capture_id += 1
                
    windows_arr = np.concatenate(windows_list, axis=0) if windows_list else np.zeros((0, 150, 256), dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)
    capture_arr = np.array(capture_ids, dtype=np.int64)
    files_arr = np.array(file_names, dtype=str)
    rooms_arr = np.zeros(len(labels_arr), dtype=np.int64) # 0 = Room A
    
    return windows_arr, labels_arr, capture_arr, rooms_arr, files_arr

def process_room_b(files_list, is_val=False):
    windows_list = []
    labels_list = []
    capture_ids = []
    file_names = []
    capture_id = 0
    
    print(f"Processing Room B ({'validation' if is_val else 'development'})...")
    for f, prefix in tqdm(files_list, desc="Room B captures"):
        try:
            with open(f, 'r') as fp:
                first_line = fp.readline()
            if 'Subcarrier' in first_line or not first_line.split(',')[0].strip().replace('.', '', 1).isdigit():
                df = pd.read_csv(f, header=0)
            else:
                df = pd.read_csv(f, header=None)
            amp = df.values.astype(np.float32)
            wins = window_csi_sequence(amp, window_size=150)
            if len(wins) > 0:
                act_id = ACTIVITY_TO_ID[ROOM_B_NAME_MAP[prefix]]
                windows_list.append(wins)
                labels_list.extend([act_id] * len(wins))
                capture_ids.extend([capture_id] * len(wins))
                file_names.extend([os.path.basename(f)] * len(wins))
                capture_id += 1
        except Exception as e:
            print(f"Error processing {f}: {e}")
            continue
            
    windows_arr = np.concatenate(windows_list, axis=0) if windows_list else np.zeros((0, 150, 256), dtype=np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)
    capture_arr = np.array(capture_ids, dtype=np.int64)
    files_arr = np.array(file_names, dtype=str)
    rooms_arr = np.ones(len(labels_arr), dtype=np.int64) # 1 = Room B
    
    return windows_arr, labels_arr, capture_arr, rooms_arr, files_arr

def build_room_b_splits(room_b_path, seed=42, val_size=25):
    csv_files = glob.glob(os.path.join(room_b_path, "*.csv"))
    class_groups = {}
    for f in csv_files:
        prefix = os.path.basename(f).split('_')[0]
        if prefix in ROOM_B_NAME_MAP:
            class_groups.setdefault(prefix, []).append(f)
            
    rng = np.random.RandomState(seed)
    train_files = []
    val_files = []
    
    for prefix in sorted(class_groups.keys()):
        files = sorted(class_groups[prefix])
        indices = rng.permutation(len(files))
        val_idx = set(indices[:val_size])
        
        for idx, f in enumerate(files):
            if idx in val_idx:
                val_files.append((f, prefix))
            else:
                train_files.append((f, prefix))
                
    return train_files, val_files

def main():
    out_dir = r"D:\ubi\p2\processed_data"
    os.makedirs(out_dir, exist_ok=True)
    
    room_a_path = r"D:\ubi\p2\data\Experiment-2"
    room_b_path = r"D:\ubi\p2\data\Experiment-3"
    
    t0 = time.time()
    
    # 1. Process Room A Dev
    a_dev_file = os.path.join(out_dir, "room_a_dev.npz")
    if not os.path.exists(a_dev_file):
        a_dev_w, a_dev_y, a_dev_c, a_dev_r, a_dev_f = process_room_a(room_a_path, is_val=False)
        np.savez_compressed(a_dev_file, windows=a_dev_w, labels=a_dev_y, capture_ids=a_dev_c, rooms=a_dev_r, files=a_dev_f)
        print(f"Room A Dev saved: {a_dev_w.shape[0]} windows, shape={a_dev_w.shape}")
    else:
        print("Room A Dev already cached, skipping.")
        
    # 2. Process Room A Val
    a_val_file = os.path.join(out_dir, "room_a_val.npz")
    if not os.path.exists(a_val_file):
        a_val_w, a_val_y, a_val_c, a_val_r, a_val_f = process_room_a(room_a_path, is_val=True)
        np.savez_compressed(a_val_file, windows=a_val_w, labels=a_val_y, capture_ids=a_val_c, rooms=a_val_r, files=a_val_f)
        print(f"Room A Val saved: {a_val_w.shape[0]} windows, shape={a_val_w.shape}")
    else:
        print("Room A Val already cached, skipping.")
    
    # 3. Process Room B
    b_train_files, b_val_files = build_room_b_splits(room_b_path, seed=42, val_size=25)
    
    b_dev_w, b_dev_y, b_dev_c, b_dev_r, b_dev_f = process_room_b(b_train_files, is_val=False)
    np.savez_compressed(os.path.join(out_dir, "room_b_dev.npz"),
                        windows=b_dev_w, labels=b_dev_y, capture_ids=b_dev_c, rooms=b_dev_r, files=b_dev_f)
    print(f"Room B Dev saved: {b_dev_w.shape[0]} windows, shape={b_dev_w.shape}")
    
    b_val_w, b_val_y, b_val_c, b_val_r, b_val_f = process_room_b(b_val_files, is_val=True)
    np.savez_compressed(os.path.join(out_dir, "room_b_val.npz"),
                        windows=b_val_w, labels=b_val_y, capture_ids=b_val_c, rooms=b_val_r, files=b_val_f)
    print(f"Room B Val saved: {b_val_w.shape[0]} windows, shape={b_val_w.shape}")
    
    print(f"\nAll data successfully processed and cached in {time.time() - t0:.2f} seconds!")

if __name__ == "__main__":
    main()
