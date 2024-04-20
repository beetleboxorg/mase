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
    self.sum_scaled_metrics = self.config["setup"]["sum_scaled_metrics"]
    self.metric_names = list(sorted(self.config["metrics"].keys()))
    if not self.sum_scaled_metrics:
        self.directions = [
            self.config["metrics"][k]["direction"] for k in self.metric_names
        ]
    else:
        self.direction = self.config["setup"]["direction"]

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

  def save_the_best(self, best_trial, save_path):
      df = pd.DataFrame(
          columns=[
              "number",
              "value",
              "software_metrics",
              "hardware_metrics",
              "scaled_metrics",
              "sampled_config",
          ]
      )
      best_trial = best_trial
      row = [
          best_trial.get('best_number'),
          best_trial.get('best_value'),
          best_trial.get('best_software_metrics'),
          best_trial.get('best_hardware_metrics'),
          best_trial.get('best_scaled_metrics'),
          best_trial.get('best_sampled_config'),
      ]
      df.loc[len(df)] = row
      df.to_json(save_path, orient="index", indent=4)

      txt = "Best trial(s):\n"
      df_truncated = df.loc[
          :, ["number", "software_metrics", "hardware_metrics", "scaled_metrics"]
      ]

      def beautify_metric(metric: dict):
          beautified = {}
          for k, v in metric.items():
              if isinstance(v, (float, int)):
                  beautified[k] = round(v, 3)
              else:
                  txt = str(v)
                  if len(txt) > 20:
                      txt = txt[:20] + "..."
                  else:
                      txt = txt[:20]
                  beautified[k] = txt
          return beautified

      #df_truncated.loc[
      #    :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
      #] = df_truncated.loc[
        #   :, ["software_metrics", "hardware_metrics", "scaled_metrics"]
      #].map(
        #   beautify_metric
      #)
      txt += tabulate(
          df_truncated,
          headers="keys",
          tablefmt="orgtbl",
      )
      logger.info(f"Best trial(s):\n{txt}")
      return df



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
    best_ratio=0
    for i, sample_config in enumerate(search_spaces):
        model = search_space.rebuild_model(sample_config, is_eval_mode)
        software_metrics = self.compute_software_metrics(
            model, sample_config, is_eval_mode
        )
        hardware_metrics = self.compute_hardware_metrics(
            model, sample_config, is_eval_mode
        )
        metrics = software_metrics | hardware_metrics
        scaled_metrics = {}
        for metric_name in self.metric_names:
            scaled_metrics[metric_name] = (
                self.config["metrics"][metric_name]["scale"] * metrics[metric_name]
            )
        self.visualizer.log_metrics(metrics=scaled_metrics, step=i)

        if not self.sum_scaled_metrics:
            list(scaled_metrics.values())
        else:
            sum(scaled_metrics.values())
        ratio=scaled_metrics.get('accuracy')/scaled_metrics.get('average_bitwidth')
        scaled_metrics.update({'ratio': ratio})
        #Compare to previous best
        if ratio>best_ratio:
          best_trial={
            'best_number':i,
            'best_value': [scaled_metrics.get('accuracy'),
                          scaled_metrics.get('average_bitwidth')],
            'best_software_metrics':software_metrics,
            'best_hardware_metrics':hardware_metrics,
            'best_scaled_metrics':scaled_metrics,
            'best_sampled_config':sample_config,

          }
          best_model=model
          best_ratio=ratio
    self.save_the_best(best_trial, self.save_dir / "best.json")

 