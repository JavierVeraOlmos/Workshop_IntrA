config = {
    "tr_set": {
        "optimizer": {
            "lr": 1e-3,
            "NAME": "adam",
            "momentum": 0.9,
            "weight_decay": 1.0e-8,
        },
        "scheduler": {
            "sched": "cosine",
            "warmup_epochs": 0,
            "min_lr": 1e-6,
        },
        "loss": {
            "crossentropy": 1,
        },
    },
    "model_parameter": {
        "in_channels": 6,
        "out_channels": 2,
        "stride": [1, 2, 2, 2],
        "nsample": [4, 4, 4, 4],
        "blocks": [4, 4, 4, 4],
        "block_num": 4,
        "planes": [64, 128, 256, 512],
    },
}
