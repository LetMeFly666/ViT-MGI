class ExperimentConfig():
    """Edit experiment configs here"""
    attack_params = {
        'pelta': True,
        'attack': 'SAGA'
        }

    model_genus = {
        'vit': {'name': 'ViT-L/16'},
        'cnn': {'name': 'BiT-M-R101x3'} #  choices: 'BiT-M-R101x3', 'ResNet-164'
    }
