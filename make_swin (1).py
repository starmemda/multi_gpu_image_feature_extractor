from swin_transformer import SwinTransformer

def swin_base_patch4_window12_384(**kwargs):
    """ Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(img_size=384,
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_base_patch4_window7_224(**kwargs):
    """ Swin-B @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_large_patch4_window12_384(**kwargs):
    """ Swin-L @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(img_size=384,
        patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_large_patch4_window7_224(**kwargs):
    """ Swin-L @ 224x224, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_small_patch4_window7_224(**kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_tiny_patch4_window7_224(**kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_base_patch4_window12_384_in22k(**kwargs):
    """ Swin-B @ 384x384, trained ImageNet-22k
    """
    model_kwargs = dict(img_size=384,
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_base_patch4_window7_224_in22k(**kwargs):
    """ Swin-B @ 224x224, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_large_patch4_window12_384_in22k(**kwargs):
    """ Swin-L @ 384x384, trained ImageNet-22k
    """
    model_kwargs = dict(img_size=384,
        patch_size=4, window_size=12, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return SwinTransformer(**model_kwargs)


def swin_large_patch4_window7_224_in22k(**kwargs):
    """ Swin-L @ 224x224, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), **kwargs)
    return SwinTransformer(**model_kwargs)




def mltr_small_patch4_window7_224(**kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=98, depths=(2, 2, 6, 2), num_heads=(7, 7, 7, 7), **kwargs)
    return SwinTransformer(**model_kwargs)