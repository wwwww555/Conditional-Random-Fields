#训练一个基于 BERT 的句子分类模型，并使用交叉验证的方式进行评估，最终保存最优模型并计算评估指标。
import time    #记录训练时间
import gc     #进行垃圾回收，释放 CUDA 内存
from datetime import datetime      #记录训练时间
from os import makedirs   #用于创建目录
import torch   #负责深度学习模型的训练和保存

from eval_run import eval_and_save_metrics   #用于计算和保存模型评估指标
from utils import get_device, ResultWriter, log   #相关工具函数
from task import pubmed_task         #具体的数据集任务
from train import SentenceClassificationTrainer   #负责训练的类
from models import BertHSLN    #定义的句子分类模型（基于 BERT）
import os

# BERT_VOCAB = "bert-base-uncased"
BERT_MODEL = "bert-base-uncased"
# BERT_VOCAB = "bert_model/scibert_scivocab_uncased/vocab.txt"
#BERT_MODEL = "allenai/scibert_scivocab_uncased"


config = {
    "bert_model": BERT_MODEL,  #使用 BERT-base-uncased 作为预训练模型
    "bert_trainable": False,   #设置为 False，意味着 BERT 的权重在训练过程中不会被更新（即只作为特征提取器）
    "model": BertHSLN.__name__,
    "cacheable_tasks": [],

    "dropout": 0.5,  #防止过拟合
    "word_lstm_hs": 758,   #BiLSTM 的隐藏层维度（758）
    "att_pooling_dim_ctx": 200,  #注意力池化的上下文维度
    "att_pooling_num_ctx": 15,   #注意力池化的上下文数目
    "lr": 3e-05,                 #初始学习率 3e-5
    "lr_epoch_decay": 0.9,       #每个 epoch 结束后，学习率衰减 0.9 倍
    "batch_size":  32,
    "max_seq_length": 128,       #最大序列长度
    "max_epochs": 20,            #最大训练轮数
    "early_stopping": 5,         #如果连续 5 轮没有提升，则停止训练

}

MAX_DOCS = -1   #表示不限制数据集大小
def create_task(create_func):
    return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS)
#封装了 create_func()，用于创建一个任务对象，并传入 batch_size 和 max_docs
def create_generic_task(task_name):
    return generic_task(task_name, train_batch_size=config["batch_size"], max_docs=MAX_DOCS)

# ADAPT: Uncomment the task that has to be trained and comment all other tasks out
task = create_task(pubmed_task)   
#task = create_task(pubmed_task_small)
#task = create_task(nicta_task)
#task = create_task(dri_task)
#task = create_task(art_task)
#task = create_task(art_task_small)
#task = create_generic_task(GEN_DRI_TASK)
#task = create_generic_task(GEN_PMD_TASK)
#task = create_generic_task(GEN_NIC_TASK)
# task = create_generic_task(GEN_ART_TASK)

# ADAPT: Set to False if you do not want to save the best model
save_best_models = True  

# ADAPT: provide a different device if needed
device = get_device(0)   

timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")   #生成当前时间的字符串（格式 YYYY-MM-DD_HH_MM_SS）

# ADAPT: adapt the folder name of the run if necessary
run = f"{timestamp}_{task.task_name}_baseline"   

# -------------------------------------------

os.makedirs("results/complete_epoch_wise_new",exist_ok=True)
#run_results = f'/nfs/data/sentence-classification/results/{run}'
run_results = f'results/{run}'    #设定保存训练结果的目录
makedirs(run_results, exist_ok=False)

# preload data if not already done
task.get_folds()   

restarts = 1 if task.num_folds == 1 else 1
for restart in range(restarts):
    for fold_num, fold in enumerate(task.get_folds()):
        start = time.time()
        result_writer = ResultWriter(f"{run_results}/{restart}_{fold_num}_results.jsonl")
        result_writer.log(f"Fold {fold_num} of {task.num_folds}")
        result_writer.log(f"Starting training {restart} for fold {fold_num}... ")

        trainer = SentenceClassificationTrainer(device, config, task, result_writer)   #训练器
        best_model = trainer.run_training_for_fold(fold_num, fold, return_best_model=save_best_models)   #执行训练
        if best_model is not None:
            model_path = os.path.join(run_results, f'{restart}_{fold_num}_model.pt')
            result_writer.log(f"saving best model to {model_path}")
            torch.save(best_model.state_dict(), model_path)

        result_writer.log(f"finished training {restart} for fold {fold_num}: {time.time() - start}")
    
        # explicitly call garbage collector so that CUDA memory is released
        gc.collect()

log("Training finished.")

log("Calculating metrics...")
eval_and_save_metrics(run_results)   
log("Calculating metrics finished")
