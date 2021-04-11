from text import symbols


class HParams(object):
    hparamdict = []

    def __init__(self, **hparams):
        self.hparamdict = hparams
        for k, v in hparams.items():
            setattr(self, k, v)

    def __repr__(self):
        return "HParams(" + repr([(k, v) for k, v in self.hparamdict.items()]) + ")"

    def __str__(self):
        return ",".join([(k + "=" + str(v)) for k, v in self.hparamdict.items()])

    def parse(self, params):
        for s in params.split(","):
            k, v = s.split("=", 1)
            k = k.strip()
            t = type(self.hparamdict[k])
            if t == bool:
                v = v.strip().lower()
                if v in ["true", "1"]:
                    v = True
                elif v in ["false", "0"]:
                    v = False
                else:
                    raise ValueError(v)
            else:
                v = t(v)
            self.hparamdict[k] = v
            setattr(self, k, v)
        return self


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        fp16_run=False,
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=["embedding.weight"],
        ################################
        # Data Parameters             #
        ################################
        training_files="filelists/ljs_audio_text_train_filelist.txt",
        validation_files="filelists/ljs_audio_text_val_filelist.txt",
        text_cleaners=["english_cleaners"],
        ################################
        # Landmark Parameters             #
        ################################
        n_mel_channels=204,
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,
        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,
        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
        mask_padding=True,  # set model's padded outputs to padded values
    )

    if hparams_string:
        print("Parsing command line hparams: %s", hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print("Final parsed hparams: %s", hparams.values())

    return hparams
