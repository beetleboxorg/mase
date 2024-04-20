import torch
import pandas as pd
import logging
import copy

from tabulate import tabulate
import joblib
from torchmetrics.classification import MulticlassAccuracy
from chop.passes.graph.analysis.quantization import calculate_avg_bits_mg_analysis_pass
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)

from functools import partial
from .base import SearchStrategyBase

from chop.passes.module.analysis import calculate_avg_bits_module_analysis_pass

logger = logging.getLogger(__name__)

class SearchStrategyBruteForce(SearchStrategyBase):
  def _post_init_setup(self):
    pass


  def get_in_frac_widths(self,data_type, search_space):
    frac_width_string="%s_frac_width" % data_type
    width_string="%s_width" % data_type

    if search_space.get(frac_width_string)==[None]:
      in_frac_widths=[]
      for in_width_sample in search_space.get(width_string):
        in_frac_widths.append((in_width_sample, in_width_sample//2))
    else:
      in_frac_widths=search_space.get(frac_width_string)
    return(in_frac_widths)

  def compute_software_metrics(self, model, sampled_config: dict, is_eval_mode: bool):
    # note that model can be mase_graph or nn.Module
    metrics = {}
    if is_eval_mode:
        with torch.no_grad():
            for runner in self.sw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
    else:
        for runner in self.sw_runner:
            metrics |= runner(self.data_module, model, sampled_config)
    return metrics

  def compute_hardware_metrics(self, model, sampled_config, is_eval_mode: bool):
    metrics = {}
    if is_eval_mode:
        with torch.no_grad():
            for runner in self.hw_runner:
                metrics |= runner(self.data_module, model, sampled_config)
    else:
        for runner in self.hw_runner:
            metrics |= runner(self.data_module, model, sampled_config)
    return metrics

  def search(self, search_space):
    pass_args = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": 0,
                "data_in_frac_width": 0,
                # weight
                "weight_width": 0,
                "weight_frac_width": 0,
                # bias
                "bias_width": 0,
                "bias_frac_width": 0,
            }
    },}
    seed=search_space.config.get("seed")
    linear_search_space=seed.get("linear").get('config')
    data_in_frac_widths=self.get_in_frac_widths('data_in', linear_search_space)
    weight_frac_widths=self.get_in_frac_widths('weight', linear_search_space)
    bias_frac_widths=self.get_in_frac_widths('weight', linear_search_space)

    search_spaces = []
    for d_config in data_in_frac_widths:
        for w_config in weight_frac_widths:
          for b_config in bias_frac_widths:
             pass_args['linear']['config']['data_in_width'] = d_config[0]
             pass_args['linear']['config']['data_in_frac_width'] = d_config[1]
             pass_args['linear']['config']['weight_width'] = w_config[0]
             pass_args['linear']['config']['weight_frac_width'] = w_config[1]
             pass_args['linear']['config']['bias_width'] = b_config[0]
             pass_args['linear']['config']['bias_frac_width'] = b_config[1]
             # dict.copy() and dict(dict) only perform shallow copies
             # in fact, only primitive data types in python are doing implicit copy when a = b happens
             search_spaces.append(copy.deepcopy(pass_args))
              
    is_eval_mode = self.config.get("eval_mode", True)
    model = search_space.rebuild_model(sampled_config, is_eval_mode)
      
    software_metrics = self.compute_software_metrics(
        model, sampled_config, is_eval_mode
    )
    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, None)

    metric = MulticlassAccuracy(num_classes=5)
    num_batchs = 5
    recorded_accs = []
    for i, config in enumerate(search_spaces):
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
