#!/usr/bin/env/python3
"""Recipe for training a sequence-to-sequence ASR system with librispeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/train_BPE1000.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
language model is used  on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.

This recipe assumes that the tokenizer and the LM are already trained.
To avoid token mismatches, the tokenizer used for the acoustic model is
the same use for the LM.  The recipe downloads the pre-trained tokenizer
and LM.

If you would like to train a full system from scratch do the following:
1- Train a tokenizer (see ../../Tokenizer)
2- Train a language model (see ../../LM)
3- Train the acoustic model (with this code).



Authors
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.nnet.losses import compute_masked_loss
from hyperpyyaml import load_hyperpyyaml


logger = logging.getLogger(__name__)


def kl_div(mu, sigma):
    return 0.5 * (torch.exp(sigma) + mu ** 2 - 1 - sigma)


# Define training procedure
class VCBrain(sb.Brain):
    def compute_feature(self, wavs, wav_lens):
        stft = self.hparams.compute_STFT(wavs)
        mag = self.hparams.spec_mag(stft)
        feats = self.hparams.fbank(mag)
        feats = self.modules.normalize(feats, wav_lens)
        return feats

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if stage != sb.Stage.TEST:
            batch = batch.to(self.device)
            wavs, wav_lens = batch.sig
            feats = self.compute_feature(wavs, wav_lens)

            # Forward pass
            x, mu, sigma, con, spk = self.modules.model(
                feats.detach(), feats.detach()
            )
            return x, mu, sigma, feats
        else:
            batch = batch.to(self.device)
            src_wavs, src_wav_lens = batch.src_sig
            tar_wavs, tar_wav_lens = batch.tar_sig
            src_feats = self.compute_feature(src_wavs, src_wav_lens)
            tar_feats = self.compute_feature(tar_wavs, tar_wav_lens)

            # Forward pass
            x, *_ = self.modules.model(
                src_feats.detach(), tar_feats.detach(), noise=False
            )

            # Rebuild wavform
            stat_dict = self.modules.normalize._statistics_dict()
            x = x * stat_dict["glob_std"] + stat_dict["glob_mean"]
            spectrograms = self.hparams.invert_fbank(x)
            # predicted_wav = self.hparams.resynth(spectrograms, src_wavs)
            predicted_wav = self.hparams.griffinlim(
                spectrograms.transpose(1, 2)
            )

            return x, src_feats, predicted_wav, src_wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the L1 loss given predictions and targets."""
        batch = batch.to(self.device)
        ids = batch.id
        if stage != sb.Stage.TEST:
            wavs, wav_lens = batch.sig
            output, mu, sigma, feats = predictions

            l1_loss = self.hparams.l1_cost(output, feats, length=wav_lens)
            kl_loss = compute_masked_loss(kl_div, mu, sigma, length=wav_lens)

            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.anneal_epochs:
                kl_weight = (
                    current_epoch / self.hparams.anneal_epochs
                ) * self.hparams.kl_weight
            else:
                kl_weight = self.hparams.kl_weight

            loss = l1_loss + kl_weight * kl_loss
            self.l1_loss_collection.append(l1_loss)
            self.kl_loss_collection.append(kl_loss)
            return loss

        else:
            x, src_feats, wavs, wav_lens = predictions
            l1_loss = self.hparams.l1_cost(x, src_feats, length=wav_lens)
            lens = wav_lens * wavs.shape[1]
            for name, wav, length in zip(ids, wavs, lens):
                vc_path = os.path.join(self.hparams.vc_folder, name)
                if not vc_path.endswith(".wav"):
                    vc_path = vc_path + ".wav"
                torchaudio.save(
                    vc_path,
                    torch.unsqueeze(wav[: int(length)].cpu(), 0),
                    self.hparams.sample_rate,
                )
            return l1_loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
            return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.l1_loss_collection = []
        self.kl_loss_collection = []

        if stage == sb.Stage.TEST:
            self.hparams.griffinlim = self.hparams.griffinlim.to(self.device)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        stage_stats = {"loss": stage_loss}
        # Compute/store important stats
        if stage != sb.Stage.TEST:
            l1_loss = (
                torch.Tensor(self.l1_loss_collection)
                .to(self.device)
                .mean()
                .item()
            )
            kl_loss = (
                torch.Tensor(self.kl_loss_collection)
                .to(self.device)
                .mean()
                .item()
            )
            stage_stats["l1_loss"] = l1_loss
            stage_stats["kl_loss"] = kl_loss

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"]},
            )

        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    # Define audio piplines
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        sig = hparams["resample"](sig)
        return sig

    # Define audio piplines for testing data
    @sb.utils.data_pipeline.takes("src_wav", "tar_wav")
    @sb.utils.data_pipeline.provides("src_sig", "tar_sig")
    def test_audio_pipeline(src_wav, tar_wav):
        src_sig = sb.dataio.dataio.read_audio(src_wav)
        src_sig = hparams["resample"](src_sig)
        tar_sig = sb.dataio.dataio.read_audio(tar_wav)
        tar_sig = hparams["resample"](tar_sig)
        return src_sig, tar_sig

    # Define datasets
    train_set = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[audio_pipeline],
        output_keys=["id", "sig"],
    )

    valid_set = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[audio_pipeline],
        output_keys=["id", "sig"],
    )

    test_set = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"],
        replacements={"data_root": hparams["data_folder"]},
        dynamic_items=[test_audio_pipeline],
        output_keys=["id", "src_sig", "tar_sig"],
    )
    return train_set, valid_set, test_set


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 1.  # Dataset prep (parsing VCTK)
    from vctk_prepare import prepare_vctk  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_vctk,
        kwargs={
            "data_folder": hparams["data_folder"],
            "dev_spks": hparams["dev_spks"],
            "te_spks": hparams["te_spks"],
            "save_folder": hparams["data_folder"],
        },
    )

    # Create the folder to save enhanced files (+ support for DDP)
    try:
        # all writing command must be done with the main_process
        if sb.utils.distributed.if_main_process():
            if not os.path.isdir(hparams["vc_folder"]):
                os.makedirs(hparams["vc_folder"])
    finally:
        # wait for main_process if ddp is used
        sb.utils.distributed.ddp_barrier()

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prep(hparams)

    # Trainer initialization
    vc_brain = VCBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    vc_brain.fit(
        vc_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    test_stats = vc_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )
