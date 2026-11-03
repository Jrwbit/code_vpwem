import torch


def test_main_camera_encoder_does_not_require_wrist_inputs(monkeypatch):
    from torch import nn
    import cleandiffuser.nn_condition.multi_image_condition as module
    class TinyVision(nn.Module):
        def __init__(self):super().__init__();self.conv1=nn.Conv2d(3,16,3,padding=1)
        def forward(self,x):return self.conv1(x).mean((-1,-2))
    monkeypatch.setattr(module,'get_resnet',lambda name:TinyVision())
    for wrist in [False,True]:
        obs={'rgb':dict(shape=[3,16,20],type='rgb'),'depth':dict(shape=[1,16,20],type='depth'),'proprio':dict(shape=[10],type='low_dim')}
        if wrist:obs.update(rgb_wrist=dict(shape=[3,16,20],type='rgb'),depth_wrist=dict(shape=[1,16,20],type='depth'))
        encoder=module.MultiImageObsConditionMomart(dict(obs=obs),'resnet18',emb_dim=32,crop_shape=None,use_seq=True,keep_horizon_dims=True)
        assert encoder.key_transform_map['depth'] is encoder.key_transform_map['rgb']
        if wrist:assert encoder.key_transform_map['depth_wrist'] is encoder.key_transform_map['rgb_wrist']
        else:assert 'rgb_wrist' not in encoder.key_model_map and 'depth_wrist' not in encoder.key_transform_map
        inputs={k:torch.rand(2,2,*v['shape']) for k,v in obs.items()}
        assert encoder(inputs).shape==(2,2,32)


