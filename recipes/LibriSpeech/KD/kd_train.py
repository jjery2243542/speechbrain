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
class KD_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        "Given an input batch it computes the phoneme probabilities."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        projections = []

        with torch.no_grad():
            feats = self.hparams.wav2vec(wavs)
            feats = feats.split(self.hparams.hidden_size, dim=-1)
            for feat in feats:
                projections.append(self.hparams.projection(feat))

        out = self.modules.model(wavs)
        output_lst, clusters_lst = [], []
        for i in range(self.hparams.num_layers):
            logits = self.modules.linears[i](out[i])
            log_probs = self.hparams.log_softmax(logits / self.hparams.gamma)
            if self.hparams.soft_dist:
                clusters = self.hparams.kmeans_models[i].transform(
                    projections[i], wav_lens, gamma=self.hparams.gamma,
                )
            else:
                clusters = self.hparams.kmeans_models[i].predict(
                    projections[i], wav_lens
                )
            output_lst.append(log_probs)
            clusters_lst.append(clusters)
        return output_lst, clusters_lst, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CTC loss."
        log_probs, clusters, wav_lens = predictions
        total_loss = 0.0
        for i in range(self.hparams.num_layers):
            if self.hparams.soft_dist:
                loss = self.hparams.kl_cost(log_probs[i], clusters[i], wav_lens)
            else:
                loss = self.hparams.nll_cost(
                    log_probs[i], clusters[i], wav_lens
                )

            total_loss += loss
            self.loss_metrics[i].append(
                ids=batch.id,
                log_probabilities=log_probs[i],
                targets=clusters[i],
                length=wav_lens,
            )
            targets = (
                clusters[i].max(dim=-1)[1]
                if self.hparams.soft_dist
                else clusters[i]
            )
            self.acc_metrics[i].append(
                log_probabilities=log_probs[i],
                targets=targets,
                length=wav_lens,
            )
        return total_loss / self.hparams.num_layers

    def fit_batch(self, batch):
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not finite
            self.check_gradients(loss)
            self.hparams.lr_annealing(self.optimizer)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.acc_metrics = [
            self.hparams.acc_stats() for _ in range(self.hparams.num_layers)
        ]
        if self.hparams.soft_dist:
            self.loss_metrics = [
                self.hparams.kl_loss_stats()
                for _ in range(self.hparams.num_layers)
            ]
        else:
            self.loss_metrics = [
                self.hparams.nll_loss_stats()
                for _ in range(self.hparams.num_layers)
            ]

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_loss_lst = {
                f"loss_l{i}": self.loss_metrics[i].summarize("average")
                for i in range(self.hparams.num_layers)
            }
            self.train_acc_lst = {
                f"acc_l{i}": self.acc_metrics[i].summarize()
                for i in range(self.hparams.num_layers)
            }
            self.train_acc = (
                sum([val for key, val in self.train_acc_lst.items()])
                / self.hparams.num_layers
            )

        else:
            self.stage_loss_lst = {
                f"loss_l{i}": self.loss_metrics[i].summarize("average")
                for i in range(self.hparams.num_layers)
            }
            self.stage_acc_lst = {
                f"acc_l{i}": self.acc_metrics[i].summarize()
                for i in range(self.hparams.num_layers)
            }
            self.stage_acc = (
                sum([val for key, val in self.stage_acc_lst.items()])
                / self.hparams.num_layers
            )

        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats={
                    "total_loss": self.train_loss,
                    "total_acc": self.train_acc,
                    **self.train_loss_lst,
                    **self.train_acc_lst,
                },
                valid_stats={
                    "total_loss": stage_loss,
                    "total_acc": self.stage_acc,
                    **self.stage_loss_lst,
                    **self.stage_acc_lst,
                },
            )
            if (
                self.hparams.epoch_counter.current
                % self.hparams.save_ckpt_every_n_epochs
                == 0
            ):
                self.checkpointer.save_and_keep_only(
                    meta={"ACC": self.stage_acc}, max_keys=["ACC"],
                )


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
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides("char_list", "char_encoded")
    def text_pipeline(char):
        char_list = char.strip().split()
        yield char_list
        char_encoded = label_encoder.encode_sequence_torch(char_list)
        yield char_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-gpu dpp support)
    lab_enc_file = os.path.join(hparams["csv_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "char_encoded"])

    return train_data, valid_data, test_datasets, label_encoder


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
    train_data, valid_data, test_datasets, label_encoder = dataio_prep(hparams)

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    linears = torch.nn.ModuleList(
        [hparams["linear"]() for _ in range(hparams["num_layers"])]
    )
    hparams["modules"]["linears"] = linears
    hparams["checkpointer"].add_recoverable("linears", linears)

    # convs = torch.nn.ModuleList([hparams["conv"]() for _ in range(hparams["num_layers"])])
    # hparams["modules"]["convs"] = convs
    # hparams["checkpointer"].add_recoverable("convs", convs)

    # Fix the projection linear layer
    for params in hparams["projection"].parameters():
        params.requires_grad = False

    # Trainer initialization
    kd_brain = KD_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    kd_brain.label_encoder = label_encoder
    # Training/validation loop
    kd_brain.fit(
        kd_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        kd_brain.evaluate(
            test_datasets[k],
            max_key="ACC",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
