"""Read-only training view for either MoMaRT cache format."""
import json
import os
from collections import OrderedDict
import numpy as np
from cleandiffuser.dataset.replay_buffer import ReplayBuffer


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


class TrainingArray:
    """Translate logical training frame indices without copying image data."""
    def __init__(self, array, frame_map):
        self.array = array
        self.frame_map = frame_map
        self.shape = (len(frame_map),) + array.shape[1:]
        self.dtype = array.dtype
        self._frame_chunks = OrderedDict()
        self._frame_chunk_bytes = 0
        self._cache_pid = os.getpid()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_frame_chunks'] = OrderedDict()
        state['_frame_chunk_bytes'] = 0
        return state

    def read_frames(self, indices):
        """Gather exact pixels, decompressing each touched Zarr chunk once.

        Keep a bounded worker-local cache for overlapping history windows.
        The budget is per observation array; no processed images or features
        are cached, so augmentation and encoder gradients remain unchanged.
        """
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError('frame indices must be one-dimensional')
        physical = self.frame_map[indices]
        result = np.empty((len(indices),) + self.shape[1:], dtype=self.dtype)
        if not len(indices):
            return result
        if self._cache_pid != os.getpid():
            self._frame_chunks.clear()
            self._frame_chunk_bytes = 0
            self._cache_pid = os.getpid()
        chunks = getattr(self.array, 'chunks', None)
        if chunks is None:
            # HDF5 original-pixel fallback and in-memory arrays still avoid
            # reading the padding frame repeatedly.
            unique, inverse = np.unique(physical, return_inverse=True)
            frames = np.concatenate([self.array[int(i):int(i)+1] for i in unique])
            return frames[inverse]
        chunk_length = int(chunks[0])
        chunk_ids = physical // chunk_length
        budget = max(0, int(float(os.environ.get('VPWEM_HISTORY_CACHE_MIB', '32')) * 2**20))
        for chunk_id in np.unique(chunk_ids):
            chunk_id = int(chunk_id)
            frames = self._frame_chunks.pop(chunk_id, None)
            if frames is None:
                start = chunk_id * chunk_length
                frames = np.asarray(self.array[start:min(start+chunk_length, len(self.array))])
                if frames.nbytes <= budget:
                    while self._frame_chunks and self._frame_chunk_bytes + frames.nbytes > budget:
                        _, old = self._frame_chunks.popitem(last=False)
                        self._frame_chunk_bytes -= old.nbytes
                    self._frame_chunk_bytes += frames.nbytes
                    self._frame_chunks[chunk_id] = frames
            else:
                self._frame_chunks[chunk_id] = frames
            positions = np.flatnonzero(chunk_ids == chunk_id)
            result[positions] = frames[physical[positions] - chunk_id * chunk_length]
        return result

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.array[int(self.frame_map[key])]
        if not isinstance(key, slice):
            raise TypeError("TrainingArray supports scalar and slice reads only")
        indices = self.frame_map[key]
        if not len(indices):
            return np.empty((0,) + self.shape[1:], dtype=self.dtype)
        # Batch adjacent physical frames; sequence reads remain one Zarr slice.
        cuts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1, len(indices)]
        pieces = [self.array[int(indices[a]):int(indices[b-1])+1]
                  for a, b in zip(cuts[:-1], cuts[1:])]
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)


class Hdf5ImageArray:
    """Original pixels for full caches containing lossy image compression."""
    def __init__(self, path, key, names, ends, shape, dtype):
        self.path, self.key = path, key
        self.names, self.ends = names, ends
        self.shape, self.dtype = shape, dtype
        self._handle = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_handle'] = None
        state['_pid'] = None
        return state

    def __len__(self):
        return self.shape[0]

    def _file(self):
        import h5py
        if self._handle is None or self._pid != os.getpid():
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(self.path, 'r')
            self._pid = os.getpid()
        return self._handle

    def __getitem__(self, key):
        scalar = isinstance(key, (int, np.integer))
        if scalar:
            index = int(key)
            if index < 0:
                index += len(self)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            start, stop = index, index+1
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step != 1:
                raise ValueError('HDF5 image reads require contiguous slices')
        else:
            raise TypeError('Scalar or slice required')
        pieces = []
        while start < stop:
            episode = int(np.searchsorted(self.ends, start, side='right'))
            origin = int(self.ends[episode-1]) if episode else 0
            end = min(stop, int(self.ends[episode]))
            array = self._file()['data/'+self.names[episode]+'/obs/'+self.key]
            if array.shape[0] != int(self.ends[episode])-origin or array.shape[1:] != self.shape[1:]:
                raise ValueError('HDF5/cache episode shape mismatch')
            pieces.append(np.asarray(array[start-origin:end-origin], dtype=self.dtype))
            start = end
        result = (np.concatenate(pieces) if pieces else
                  np.empty((0,)+self.shape[1:], dtype=self.dtype))
        return result[0] if scalar else result


def training_replay(root):
    if root.attrs.get("vpwem_cache_complete") is not True:
        raise ValueError("Incomplete MoMaRT cache")
    kind = root.attrs.get("vpwem_cache_kind", root.attrs.get("dataset_kind"))
    if kind != "momart":
        raise ValueError("Wrong MoMaRT cache kind")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if not len(ends) or np.any(np.diff(np.r_[0, ends]) <= 0):
        raise ValueError("Invalid episode boundaries")
    if root.attrs.get("vpwem_cache_kind") == "momart":
        names = decoded(root.attrs["demo_names"])
        selected = np.arange(len(names))
    else:
        names = decoded(root.attrs["source_demos"])
        masks = decoded(root.attrs["source_masks"])
        if "train" not in masks:
            raise ValueError("Full MoMaRT cache must have a train mask")
        selected = np.asarray(masks["train"], dtype=np.int64)
        valid = np.asarray(masks.get("valid", []), dtype=np.int64)
        if np.intersect1d(selected, valid).size:
            raise ValueError("Train/valid overlap")
    if len(names) != len(ends) or len(set(names)) != len(names):
        raise ValueError("Missing or duplicate demo identities")
    if not len(selected) or len(np.unique(selected)) != len(selected) or np.any(selected < 0) or np.any(selected >= len(names)):
        raise ValueError("Invalid training mask")
    starts = np.r_[0, ends[:-1]]
    frame_map = np.concatenate([np.arange(starts[i], ends[i], dtype=np.int64)
                                for i in selected])
    train_ends = np.cumsum((ends-starts)[selected], dtype=np.int64)
    data = {}
    for key, array in root["data"].items():
        if len(array) != int(ends[-1]):
            raise ValueError("Source array length mismatch: " + key)
        compressor = getattr(array, 'compressor', None)
        config = compressor.get_config() if compressor else {}
        if (root.attrs.get('dataset_kind') == 'momart' and
                config.get('id') in ('imagecodecs_jpegxl', 'imagecodecs_jpeg2k')):
            source = root.attrs.get('source_path')
            if not source or not os.path.isfile(source):
                raise ValueError('Original HDF5 required for original image pixels')
            array = Hdf5ImageArray(source, key, names, ends, array.shape, array.dtype)
        data[key] = TrainingArray(array, frame_map)
    return (ReplayBuffer({"data": data, "meta": {"episode_ends": train_ends}}),
            [names[i] for i in selected], frame_map)
