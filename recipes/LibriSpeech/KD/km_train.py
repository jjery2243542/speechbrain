#!/usr/bin/env python3
"""Recipe for training a phoneme recognizer on TIMIT.
The system relies on a model trained with CTC.
Greedy search is using for validation, while beamsearch
is used at test time to improve the system performance.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/TIMIT

Authors
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


# Define training procedure
class KM_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        phns, phn_lens = batch.phn_encoded

        # To prevent length difference
        lens = phn_lens if stage == sb.Stage.TEST else wav_lens
        with torch.no_grad():
            feats = self.hparams.wav2vec(wavs)
            feats = feats.split(self.hparams.hidden_size, dim=-1)
        projections = []
        for i, model in enumerate(self.hparams.kmeans_models):
            projection = self.hparams.linear(feats[i])
            projections.append(projection)

            if stage == sb.Stage.TRAIN:
                model(projection, lens)

        return projections, lens

    def compute_objectives(self, predictions, batch, stage):
        projections, lens = predictions
        score = 0
        for i, model in enumerate(self.hparams.kmeans_models):
            score += model.score(projections[i], lens)
        score = score / len(self.hparams.kmeans_models)
        if stage == sb.Stage.TEST:
            batch = batch.to(self.device)
            phns, phn_lens = batch.phn_encoded
            for i in range(len(self.hparams.kmeans_models)):
                clusters = self.hparams.kmeans_models[i].predict(
                    projections[i], lens
                )
                self.confusion_metrics[i].append(
                    clusters=clusters,
                    targets=phns,
                    length=lens,
                    masked_indices=self.skip_token_indices,
                )
        return score

    def fit_batch(self, batch):
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        score = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        return score

    def evaluate_batch(self, batch, stage):
        outputs = self.compute_forward(batch, stage)
        score = self.compute_objectives(outputs, batch, stage)
        return score

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        if stage == sb.Stage.TEST:
            self.confusion_metrics = [
                self.hparams.confusion_stats()
                for _ in range(len(self.hparams.return_nth_layers))
            ]

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_score = stage_loss

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats={"score": self.train_score},
                valid_stats={"score": stage_loss},
            )
            if (
                self.hparams.epoch_counter.current
                % self.hparams.save_ckpt_every_n_epochs
                == 0
            ):
                self.checkpointer.save_and_keep_only(
                    meta={"score": stage_loss}, max_keys=["score"],
                )
        if stage == sb.Stage.TEST:
            for i in range(len(self.hparams.kmeans_models)):
                clu_pur, phn_pur = self.confusion_metrics[i].summarize()
                layer_id = self.hparams.return_nth_layers[i]
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "Epoch loaded": self.hparams.epoch_counter.current,
                        "Layer": layer_id,
                    },
                    test_stats={
                        "score": stage_loss,
                        "cluster_purity": clu_pur,
                        "phone_purity": phn_pur,
                    },
                )
                confusion_file = os.path.join(
                    self.hparams.save_folder, f"confusion_table_{layer_id}.pkl"
                )
                self.confusion_metrics[i].save_confusion(confusion_file)


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        hparams["train_dataloader_opts"]["shuffle"] = True

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    label_encoder = sb.dataio.encoder.TextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-gpu dpp support)
    label_encoder.update_from_didataset(train_data, output_key="phn_list")
    skip_token_indices = [
        label_encoder.encode_label(token) for token in hparams["skip_tokens"]
    ]

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

    return (
        train_data,
        valid_data,
        test_datasets,
        label_encoder,
        skip_token_indices,
    )


# Begin Recipe!
if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset prep (parsing TIMIT and annotation into csv files)
    from librispeech_prepare import prepare_librispeech  # noqa

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["csv_folder"],
            "skip_prep": hparams["skip_prep"],
            "sample_subsets": hparams["sample_subsets"],
        },
    )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    (
        train_data,
        valid_data,
        test_datasets,
        label_encoder,
        skip_token_indices,
    ) = dataio_prep(hparams)

    # Trainer initialization
    km_brain = KM_Brain(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    km_brain.label_encoder = label_encoder
    km_brain.skip_token_indices = skip_token_indices

    lab_enc_path = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.save(lab_enc_path)

    # Training/validation loop
    km_brain.fit(
        km_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        km_brain.evaluate(
            test_datasets[k],
            max_key="score",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
