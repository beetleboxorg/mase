import torch
import pandas as pd
import logging
from tabulate import tabulate
import joblib

from functools import partial
from .base import SearchStrategyBase

from chop.passes.module.analysis import calculate_avg_bits_module_analysis_pass

logger = logging.getLogger(__name__)

class SearchStrategyBruteForce(SearchStrategyBase):
  def _post_init_setup(self):
    pass

  def search(self, search_space):
    recorded_accs = []
    for i, config in enumerate(search_spaces):
        mg, _ = quantize_transform_pass(mg, config)
        j = 0
        from chop.passes.graph import report_node_shape_analysis_pass, report_node_meta_param_analysis_pass, report_node_hardware_type_analysis_pass, report_node_type_analysis_pass
        #mg, _ = report_node_type_analysis_pass(mg)
        from chop.passes.graph.analysis.quantization import calculate_avg_bits_mg_analysis_pass
        mg, info = calculate_avg_bits_mg_analysis_pass(mg, pass_args)
        print(info)
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
