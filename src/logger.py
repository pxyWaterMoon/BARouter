from __future__ import annotations
import os
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import Self  # Python 3.11+
import matplotlib.pyplot as plt
import logging
import shutil
import torch.utils.tensorboard as tensorboard
import torch
import numpy as np
import json

class Logger:

    _instance: ClassVar[Logger] = None  # singleton pattern
    writer: tensorboard.SummaryWriter
    history: dict = {}
    log_dir: str | os.PathLike | None

    def __new__(
            cls, 
            log_dir: str | os.PathLike | None = None,
    ):      
        # if the log_dir is existed remove it
        if os.path.exists(log_dir):
            logging.warning(f"{log_dir} is exstied, remove the original dir.")
            shutil.rmtree(log_dir)
        if cls._instance is None:
            self = cls._instance = super().__new__(
                cls,
            )
            self.writer = tensorboard.SummaryWriter(log_dir=log_dir)
            self.log_dir = log_dir
        else:
            assert log_dir is None, "Cannot change log_dir after Logger is initialized"
        return cls._instance
    
    def log_scalar(self, metrics: dict[str, Any], step: int = 0):
        """
        Log metrics to TensorBoard.
        
        Args:
            metrics (dict): Dictionary of metrics to log.
            step (int): Global step value to record.
        """
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)


    def log_signal(self, prompts, actions, rewards, costs, step):
        if "prompts" not in self.history.keys():
            self.history["prompts"] = []
        if "actions" not in self.history.keys():
            self.history["actions"] = []
        if "rewards" not in self.history.keys():
            self.history["rewards"] = []
        if "costs" not in self.history.keys():
            self.history["costs"] = []
        if "global_step" not in self.history.keys():
            self.history["global_step"] = []
        self.history["prompts"].append(prompts)
        self.history["actions"].append(actions)
        self.history["rewards"].append(rewards)
        self.history["costs"].append(costs)
        self.history["global_step"].append(step)
    
    def get_log_value(self, key: str, step: range | int) -> Any:
        """
        Get the logged value for a specific key at a given step.
        
        Args:
            key (str): The key to retrieve the value for.
            step (int): The step at which the value was logged.
        
        Returns:
            Any: The logged value or None if not found.
        """
        if key not in self.history:
            raise KeyError(f"Key '{key}' not found in history.")
        log_value = self.history[key]
        values = []
        if isinstance(step, range):
            for s in step:
                values += log_value[s]
        elif isinstance(step, int):
            values = log_value[step]
        else:
            raise TypeError("Step must be an integer or a range.")
        return sum(values) / len(values)
    
    # def plot_action_log(self):
    #     fig = plt.figure(figsize=(5, 4))
    #     ax = fig.add_axes([0.12, 0.1, 0.85, 0.8])
    #     # 统计每种action的数量
    #     action_counts = {}
    #     for action in self.history["actions"]:
    #         if action not in action_counts:
    #             action_counts[action] = 0
    #         action_counts[action] += 1
    #     actions = list(action_counts.keys())
    #     counts = list(action_counts.values())
    #     ax.bar(actions, counts)
    #     # ax.set_xticks(actions, rotation=45, ha='right')
    #     ax.set_xlabel("Actions")
    #     ax.set_ylabel("Counts")
    #     ax.set_title("Action Counts")
    #     plt.savefig(os.path.join(self.log_dir, "ActionCounts.png"))
    #     plt.close(fig)
    #     self.writer.add_image("ActionCounts", torch.tensor(plt.imread(os.path.join(self.log_dir, "ActionCounts.png"))))
    #     # plt.close(fig)

    def plot_action_log(self):
        fig = plt.figure(figsize=(8, 5))  # 增加图形宽度
        ax = fig.add_subplot(111)
        
        # 统计每种action的数量
        action_counts = {}
        actions = []
        for batch in self.history["actions"]:
            actions += batch
        for action in actions:
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1
        
        # 按数量排序使图形更直观
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        actions = [a[0] for a in sorted_actions]
        counts = [a[1] for a in sorted_actions]
        
        # 使用不同颜色增强区分度
        colors = plt.cm.tab20(np.linspace(0, 1, len(actions)))
        ax.bar(actions, counts, color=colors)
        # 在bar上显示具体数值
        for i, count in enumerate(counts):
            ax.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom', fontsize=8)
        # 设置x轴标签旋转45度并右对齐
        plt.xticks(rotation=45, ha='right')
        ax.set_xlabel("Actions")
        ax.set_ylabel("Counts")
        ax.set_title("Action Counts")
        
        # 增加底部边距防止标签被裁剪
        plt.subplots_adjust(bottom=0.35)
        
        # 自动调整布局
        fig.tight_layout()

        self.writer.add_figure("results/ActionCounts", fig)
        
        # save_path = os.path.join(self.log_dir, "ActionCounts.png")
        # plt.savefig(save_path)
        # plt.close(fig)
        
        # # 修复图像格式: [H, W, C] -> [C, H, W]
        # image_array = plt.imread(save_path)
        # if image_array.ndim == 3:  # 确保是RGB/RGBA图像
        #     tensor_image = torch.tensor(image_array).permute(2, 0, 1)
        # else:  # 如果是灰度图
        #     tensor_image = torch.tensor(image_array).unsqueeze(0)
        
        # self.writer.add_image("results/ActionCounts", tensor_image)
    
    def save_history(self):
        """
        Save the history to a JSON file.
        
        Args:
            file_name (str): The name of the file to save the history.
        """
        file_name = os.path.join(self.log_dir, "history.json")
        with open(file_name, "w") as f:
            json.dump(self.history, f, indent=4)
