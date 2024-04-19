import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib
from torchmetrics.classification import MulticlassAccuracy
from chop.passes.graph.analysis.quantization import calculate_avg_bits_mg_analysis_pass
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)

from functools import partial
from .base import SearchStrategyBase

from chop.passes.module.analysis import calculate_avg_bits_module_analysis_pass

logger = logging.getLogger(__name__)

class SearchStrategyBruteForce(SearchStrategyBase):
  def _post_init_setup(self):
    pass

  def search(self, search_space):
    print(search_space)
    print(search_space.config.get("seed"))
    search_spaces = []
    for d_config in data_in_frac_widths:
        for w_config in w_in_frac_widths:
            pass_args['linear']['config']['data_in_width'] = d_config[0]
            pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
            pass_args['linear']['config']['weight_width'] = w_config[0]
            pass_args['linear']['config']['weight_frac_width'] = w_config[1]
            # dict.copy() and dict(dict) only perform shallow copies
            # in fact, only primitive data types in python are doing implicit copy when a = b happens
            search_spaces.append(copy.deepcopy(pass_args))
    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, None)

    metric = MulticlassAccuracy(num_classes=5)
    num_batchs = 5
    recorded_accs = []
    for i, config in enumerate(search_space):
        mg, _ = quantize_transform_pass(mg, config)
        j = 0
        #mg, _ = report_node_type_analysis_pass(mg)
        mg, info = calculate_avg_bits_mg_analysis_pass(mg, pass_args)
        #mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("all",)})
        # this is the inner loop, where we also call it as a runner.
        acc_avg, loss_avg = 0, 0
        accs, losses = [], []
        for inputs in data_module.train_dataloader():
            xs, ys = inputs
            preds = mg.model(xs)
            loss = torch.nn.functional.cross_entropy(preds, ys)
            acc = metric(preds, ys)
            accs.append(acc)
            losses.append(loss)
            if j > num_batchs:
                break
            j += 1
        acc_avg = sum(accs) / len(accs)
        loss_avg = sum(losses) / len(losses)
        recorded_accs.append([acc_avg, info.get('data_avg_bit'), info.get('w_avg_bit')])
        return(recorded_accs)
