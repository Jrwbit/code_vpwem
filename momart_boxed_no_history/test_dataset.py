"""Check real cached episode boundaries and training inputs on BHR."""
import unittest
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import robomimic.utils.obs_utils as ObsUtils
from .dataset import BoxedNoHistoryDataset
from .painting import draw_bbox
from .train import initialize_modalities

class DatasetTests(unittest.TestCase):
    def test_all_tasks_and_episode_end_alignment(self):
        config=OmegaConf.load(Path(__file__).with_name("config_bs256_150k.yaml"))
        if not Path(config.data_root).is_dir():
            self.skipTest("BHR MoMaRT caches are unavailable")
        initialize_modalities(config)
        for spec in config.task_datasets:
            with self.subTest(dataset=spec.name):
                dataset=BoxedNoHistoryDataset(
                    Path(config.data_root)/(spec.name+".zarr"),
                    config.shape_meta["obs"],config.horizon,config.obs_steps,
                    config.dataset_memory_subsample_ratio,
                    Path(config.bbox_root)/(spec.name+".bbox.zarr"))
                ends=dataset.replay_buffer.episode_ends[:]
                for index in [0,100,int(ends[0])-1,int(ends[0])]:
                    sample=dataset.sampler.sample_sequence(index)
                    result=dataset[index]
                    np.testing.assert_array_equal(result["actions"],sample["action"])
                    plain=ObsUtils.process_obs_dict({
                        key:sample[key][:config.obs_steps] for key in dataset.obs_keys})
                    for key in dataset.obs_keys:
                        if key!="rgb":
                            np.testing.assert_array_equal(result["obs"][key],plain[key])
                    start,end,_,_=dataset.sampler.indices[index]
                    indices=np.minimum(np.arange(start,start+config.obs_steps),end-1)
                    boxed=np.stack([draw_bbox(sample["rgb"][j],
                        dataset.bbox["boxes"][int(frame)],bool(dataset.bbox["valid"][int(frame)]))
                        for j,frame in enumerate(indices)])
                    expected=ObsUtils.process_obs_dict({"rgb":boxed})["rgb"]
                    np.testing.assert_array_equal(result["obs"]["rgb"],expected)
                    if index==int(ends[0])-1:
                        np.testing.assert_array_equal(result["obs"]["rgb"][0],result["obs"]["rgb"][1])

if __name__=="__main__": unittest.main()
