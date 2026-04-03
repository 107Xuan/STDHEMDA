import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from Config import get_config
from model import MSFEICL
from datalord import load_train_test_rigorous
from clac_metric import get_metric_best_threshold, print_metrics


def train_one_epoch(model, train_loader, optimizer, criterion, args, epoch=0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        try:
            batch_data = batch_data.to(args.device)
            batch_labels = batch_labels.to(args.device)

            #添加调试信息
            if batch_idx == 0 and epoch == 0:
                print(f"model device: {next(model.parameters()).device}")

            # 前向传播
            predictions = model(batch_data, batch_labels)

            #检查输出设备
            if batch_idx == 0 and epoch == 0:
                print(f"predictions requires_grad: {predictions.requires_grad}")

            # 计算预测损失
            prediction_loss = criterion(predictions, batch_labels)
            total_loss_batch = prediction_loss

            #检查损失
            if batch_idx == 0 and epoch == 0:
                print(f"prediction_loss: {prediction_loss.item():.4f}")

            # 反向传播
            optimizer.zero_grad()
            total_loss_batch.backward()

            # 检查梯度
            if batch_idx == 0 and epoch == 0:
                has_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        has_grad = True
                        break
                print(f"Gradients computed: {has_grad}")

            optimizer.step()

            # 累计损失
            total_loss += prediction_loss.item() * len(batch_data)
            total_samples += len(batch_data)

        except RuntimeError as e:
            print(f"\nError at batch {batch_idx}:")
            print(f"Error message: {str(e)}")
            print(f"batch_data shape: {batch_data.shape}")
            print(f"batch_labels shape: {batch_labels.shape}")
            raise e

    avg_loss = total_loss / total_samples

    return avg_loss


def evaluate(model, val_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(args.device)
            batch_labels = batch_labels.to(args.device)

            # 前向传播
            predictions = model(batch_data, batch_labels)

            # 计算损失
            prediction_loss = criterion(predictions, batch_labels)
            total_loss += prediction_loss.item() * len(batch_data)

            # 收集预测结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)

    # 计算评估指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    metrics, best_threshold = get_metric_best_threshold(all_labels, all_predictions, metric='f1')
    print(f"Best threshold: {best_threshold:.3f}")

    return avg_loss, metrics, all_predictions, all_labels


def train_model(model, train_loader, val_loader, args, fold_idx=0):
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # 早停机制
    best_auc = 0.0
    best_metrics = None
    best_predictions = None
    best_labels = None
    patience_counter = 0
    patience = 20

    print("\nStarting Training...\n")

    for epoch in range(args.epoch):
        epoch_start_time = time.time()

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, args, epoch
        )

        # 验证
        val_loss, val_metrics, val_predictions, val_labels = evaluate(
            model, val_loader, criterion, args
        )

        auc, aupr, accuracy, f1, recall, precision, specificity = val_metrics

        # 学习率调度
        scheduler.step(auc)

        # 早停检查
        if auc > best_auc:
            best_auc = auc
            best_metrics = val_metrics
            best_predictions = val_predictions
            best_labels = val_labels
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start_time

        # 打印进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch + 1:3d}/{args.epoch}] ({epoch_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val AUC:    {auc:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}")
            print(f"  Best AUC:   {best_auc:.4f} (patience: {patience_counter}/{patience})")

        # 早停
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("Training Completed!")

    return best_metrics, best_predictions, best_labels


def main():
    # 加载配置
    args = get_config()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("\nConfiguration Summary:")
    print(f"  Device:        {args.device}")
    print(f"  K-fold CV:     {args.kfold}")
    print(f"  Epochs:        {args.epoch}")
    print(f"  Batch size:    {args.batchSize}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dropout:       {args.fcDropout}")
    print(f"  Seed:          {args.seed}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Hidden dim:    {args.hidden_dim}")

    # 加载实体名称
    mirnas_df = pd.read_csv(args.mirna_file, header=0)
    drugs_df = pd.read_csv(args.drug_file, header=0)
    mirnas = mirnas_df.iloc[:, 0].tolist()
    drugs = drugs_df.iloc[:, 0].tolist()

    print(f"\nLoaded {len(mirnas)} miRNAs and {len(drugs)} drugs")

    # 存储所有fold的结果
    all_fold_metrics = []
    all_fold_predictions = []
    all_fold_labels = []

    # K折交叉验证
    print(f"Starting {args.kfold}-Fold Cross Validation")

    data_generator = load_train_test_rigorous(
        link_file=args.link_file,
        mirnas=mirnas,
        drugs=drugs,
        n_splits=args.kfold,
        random_state=args.seed
    )

    for fold_idx, (train_pairs, train_labels, val_pairs, val_labels) in enumerate(data_generator):
        print(f"\nFold {fold_idx + 1}/{args.kfold}")

        fold_start_time = time.time()

        # 初始化模型
        model = MSFEICL(args)

        # 构建当前fold的交互图
        print("\nBuilding interaction graph for this fold:")
        model.build_graph(train_pairs, train_labels)

        # 创建数据加载器
        train_dataset = TensorDataset(train_pairs, train_labels)
        val_dataset = TensorDataset(val_pairs, val_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batchSize,
            shuffle=True,
            drop_last=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batchSize,
            shuffle=False,
            drop_last=False
        )

        # 训练模型
        fold_metrics, fold_predictions, fold_labels = train_model(
            model, train_loader, val_loader, args, fold_idx
        )

        # 保存结果
        all_fold_metrics.append(fold_metrics)
        all_fold_predictions.append(fold_predictions)
        all_fold_labels.append(fold_labels)

        fold_time = time.time() - fold_start_time

        # 打印当前fold结果
        print(f"\nFold {fold_idx + 1} Results (time: {fold_time / 60:.2f} min):")
        print_metrics(fold_metrics, prefix="  ")

    # 计算平均结果

    metrics_array = np.array(all_fold_metrics)
    mean_metrics = metrics_array.mean(axis=0)
    std_metrics = metrics_array.std(axis=0)

    metric_names = ['AUC', 'AUPR', 'Accuracy', 'F1-Score', 'Recall', 'Precision', 'Specificity']

    print(f"\n{'Metric':<15} {'Mean':<10} {'Std':<10}")
    print("-" * 35)
    for name, mean, std in zip(metric_names, mean_metrics, std_metrics):
        print(f"{name:<15} {mean:.4f}    ±{std:.4f}")

    # 保存结果
    print("\nSaving results...")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # 保存每个fold的详细结果
    for fold_idx in range(args.kfold):
        fold_results = pd.DataFrame({
            'predictions': all_fold_predictions[fold_idx],
            'labels': all_fold_labels[fold_idx]
        })
        fold_results.to_csv(
            f'{results_dir}/fold_{fold_idx + 1}_predictions.csv',
            index=False
        )

    # 保存汇总结果
    summary_df = pd.DataFrame(all_fold_metrics, columns=metric_names)
    summary_df['Fold'] = [f'Fold {i + 1}' for i in range(args.kfold)]
    summary_df = summary_df[['Fold'] + metric_names]

    # 添加平均值和标准差
    mean_row = pd.DataFrame([['Mean'] + mean_metrics.tolist()], columns=summary_df.columns)
    std_row = pd.DataFrame([['Std'] + std_metrics.tolist()], columns=summary_df.columns)
    summary_df = pd.concat([summary_df, mean_row, std_row], ignore_index=True)

    summary_df.to_csv(f'{results_dir}/cv_summary.csv', index=False)

    print(f"Results saved to '{results_dir}/' directory")

    return mean_metrics, std_metrics


if __name__ == '__main__':
    try:
        mean_metrics, std_metrics = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback

        traceback.print_exc()