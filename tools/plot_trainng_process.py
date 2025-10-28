import json
import matplotlib.pyplot as plt
import os

# 讀取 JSON 檔案並將數據存儲在列表中
def read_json_log(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 繪製損失和指標並保存圖表
def plot_metrics(data, save_path):
    # 初始化空列表來存儲各個參數
    epochs = []
    loss = []
    loss_cls = []
    loss_bbox = []
    ap50 = []

    # 遍歷數據以提取所需的資訊
    for entry in data:
        if 'epoch' in entry:
            epochs.append(entry['epoch'])
            loss.append(entry.get('loss', None))
            loss_cls.append(entry.get('loss_cls', None))
            loss_bbox.append(entry.get('loss_bbox', None))


    # 繪製曲線
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Loss', color='b')
    plt.plot(epochs, loss_cls, label='Loss_cls', color='g')
    plt.plot(epochs, loss_bbox, label='Loss_bbox', color='r')


    # 設定圖表標題和標籤
    plt.title('Training Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # 保存圖表
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    # 指定 JSON 日誌檔案的路徑
    log_file_path = r'C:\Users\Owner\Desktop\RTMDet_RLO\run\billborad_3\p6_34\RBD_k_size=53_stage_234_D_boundary_norm\weight_bbox=2.5,weight_cls=0.1\lr=0.001\1000epoches\20250505_184149\vis_data\20250505_184149.json'
    
    # 讀取 JSON 日誌
    data = read_json_log(log_file_path)
    
    # 保存圖表的路徑
    save_directory = os.path.dirname(log_file_path)
    save_path = os.path.join(save_directory, 'training_result.png')
    
    # 繪製損失和 AP50 指標並保存圖表
    plot_metrics(data, save_path)
