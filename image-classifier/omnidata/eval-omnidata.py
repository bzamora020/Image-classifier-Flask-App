from mv3d.baselines.omnidata.modules.midas.dpt_depth import DPTDepthModel
from mv3d.eval.main import main
import torch
import numpy as np
import tqdm


def process_scene(batch, scene, dset, net):
    with torch.no_grad():
        ref_idx = torch.unique(batch.ref_src_edges[0])
        images = batch.images[ref_idx]

        max_batch_size = 10
        n_imgs = images.shape[0]
        n_batches = (n_imgs - 1) // max_batch_size + 1
        out_all = np.empty((n_imgs, *images.shape[-2:]))

        for b in tqdm.tqdm(range(n_batches)):
            img_batch = images[b*max_batch_size: (b+1)*max_batch_size]
            img_batch = img_batch[:, [2, 1, 0]]
            out = net(img_batch).clamp(min=0, max=1)
            depth_preds = out.squeeze(1).detach().cpu().numpy()
            out_all[b*max_batch_size: (b+1)*max_batch_size] = depth_preds
        return out_all, None, None


if __name__ == '__main__':
    print('Loading pre-trained model...')

    pretrained_weights_path = './pretrained_models/omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    print('loaded!')

    dset_kwargs = {
        'mean_rgb': [0.5, 0.5, 0.5],       # change this 
        'std_rgb': [0.5, 0.5, 0.5],
        'scale_rgb': 255.,
        'img_size': (384, 384)
    }
    main('omnidata', process_scene, dset_kwargs, model, depth=True, overwrite=True)
