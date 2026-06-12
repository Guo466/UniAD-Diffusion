import sqlite3, io
import numpy as np
import zlib
from tqdm import tqdm
import os, shutil

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    # zlib can be involved
    return sqlite3.Binary(zlib.compress(out.read()))

def convert_array(text):
    out = io.BytesIO(zlib.decompress(text))
    out.seek(0)
    return np.load(out,allow_pickle=True)

# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)

class dbWrapper():
    def __init__(self):
        self.con = None
        self.cur = None
        self.idx = 0
    def connect(self, db_path, read_only = True):
        self.con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cur = self.con.cursor()
        self.ro = read_only
    def append(self, cache):
        'insert from cache dict pickle'
        if self.ro:
            raise RuntimeError("DB is connected read only!")
        if self.idx == 0:
            # create table if empty (None can be store in any type)
            self.cur.execute("create table dataset (idx INTEGER PRIMARY KEY,"+
                             ",".join([f"{k} {'array' if type(v) == np.ndarray else ''}" for k,v in cache.items()])+");")
        self.cur.execute(f"insert into dataset values ({self.idx}{',?'*len(cache)})", tuple(cache.values()))
        # not using sql self-increse primary key since it starts with 1
        self.idx += 1   
    def close(self):
        'commit changes and close db'
        self.con.commit()
        self.con.close()
    def __len__(self):
        'get rows count in db'
        res = self.cur.execute("SELECT max(idx) FROM dataset")
        len, = res.fetchone()
        return len + 1 # TODO: drop idx use default rowid as primary key
    def __getitem__(self, index):
        'override operator[]'
        res = self.cur.execute("SELECT * FROM dataset where idx = ?", (index,))
        # assemble dict
        return dict(zip([d[0] for d in self.cur.description], res.fetchone()))

def load_to_memory(src, chunk_size=4096):
    print(f'Loading {src} to memory...')
    size = os.path.getsize(src)
    free = shutil.disk_usage("/dev/shm").free
    if size > free:
        raise MemoryError(f"Not enough memory: needs {size/(1024**3):.2f} GB but {free/(1024**3):.2f} GB available.")
    dst = os.path.join("/dev/shm", os.path.basename(src))
    # diff large files takes more time than overwrite
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst, \
        tqdm(total=size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        while True:
            chunk = fsrc.read(chunk_size)
            if not chunk:
                break
            fdst.write(chunk)
            pbar.update(len(chunk))