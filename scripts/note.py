import torch
from vgn.ConvONets.conv_onet import models, training
def GIGA():
    config = {
        'encoder': 'voxel_simple_local',
        'encoder_kwargs': {
            'plane_type': ['xz', 'xy', 'yz'],
            'plane_resolution': 40,
            'unet': True,
            'unet_kwargs': {
                'depth': 3,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        },
        'decoder': 'simple_local',
        'decoder_tsdf': True,
        'decoder_kwargs': {
            'dim': 3,
            'sample_mode': 'bilinear',
            'hidden_size': 32,
            'concat_feat': True
        },
        'padding': 0,
        'c_dim': 32
    }
    return get_model(config)


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['decoder']
    encoder = cfg['encoder']
    c_dim = cfg['c_dim']
    decoder_kwargs = cfg['decoder_kwargs']
    encoder_kwargs = cfg['encoder_kwargs']
    padding = cfg['padding']
    if padding is None:
        padding = 0.1
    
    # for pointcloud_crop
    try: 
        encoder_kwargs['unit_size'] = cfg['data']['unit_size']
        decoder_kwargs['unit_size'] = cfg['data']['unit_size']
    except:
        pass
    # local positional encoding
    if 'local_coord' in cfg.keys():
        encoder_kwargs['local_coord'] = cfg['local_coord']
        decoder_kwargs['local_coord'] = cfg['local_coord']
    if 'pos_encoding' in cfg:
        encoder_kwargs['pos_encoding'] = cfg['pos_encoding']
        decoder_kwargs['pos_encoding'] = cfg['pos_encoding']

    tsdf_only = 'tsdf_only' in cfg.keys() and cfg['tsdf_only']
    detach_tsdf = 'detach_tsdf' in cfg.keys() and cfg['detach_tsdf']

    if tsdf_only:
        decoders = []
    else:
        decoder_qual = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoder_rot = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=4,
            **decoder_kwargs
        )
        decoder_width = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders = [decoder_qual, decoder_rot, decoder_width]
    if cfg['decoder_tsdf'] or tsdf_only:
        decoder_tsdf = models.decoder_dict[decoder](
            c_dim=c_dim, padding=padding, out_dim=1,
            **decoder_kwargs
        )
        decoders.append(decoder_tsdf)

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None

    if tsdf_only:
        model = models.ConvolutionalOccupancyNetworkGeometry(
            decoder_tsdf, encoder, device=device
        )
    else:
        model = models.ConvolutionalOccupancyNetwork(
            decoders, encoder, device=device, detach_tsdf=detach_tsdf
        )

    return model