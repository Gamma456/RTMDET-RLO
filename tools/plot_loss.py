import matplotlib.pyplot as plt
from mmengine.hooks import Hook

class PlotMetricsHook(Hook):
    def __init__(self, save_path, update_interval=1):
        """
        Args:
            save_path (str): 圖表最終保存的完整路徑。
            update_interval (int): 每隔多少個 epoch 更新一次圖表（僅用於畫面更新）。
        """
        self.save_path = save_path
        self.update_interval = update_interval
        self.epochs = []
        self.loss = []
        self.loss_cls = []
        self.loss_bbox = []
        plt.ion()  # 開啟互動模式
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

    def after_train_epoch(self, runner):
        epoch = runner.epoch

        # 嘗試從 runner.outputs 取得數據
        outputs = None
        if hasattr(runner, 'outputs'):
            outputs = runner.outputs

        # 如果 outputs 為空或不存在，改從 message_hub 嘗試取得
        if outputs is None or outputs == {}:
            if hasattr(runner, 'message_hub'):
                outputs = {
                    'loss': runner.message_hub.get_scalar('loss'),
                    'loss_cls': runner.message_hub.get_scalar('loss_cls'),
                    'loss_bbox': runner.message_hub.get_scalar('loss_bbox')
                }
            else:
                outputs = {}

        print(f"Epoch {epoch} outputs: {outputs}")
        if (not outputs or 
            outputs.get('loss') is None or 
            outputs.get('loss_cls') is None or 
            outputs.get('loss_bbox') is None):
            print("Some metrics are missing; skipping update.")
            return

        self.epochs.append(epoch)
        self.loss.append(outputs.get('loss'))
        self.loss_cls.append(outputs.get('loss_cls'))
        self.loss_bbox.append(outputs.get('loss_bbox'))

        if epoch % self.update_interval == 0:
            self.ax.clear()
            self.ax.plot(self.epochs, self.loss, label='Loss', color='blue')
            self.ax.plot(self.epochs, self.loss_cls, label='Loss_cls', color='green')
            self.ax.plot(self.epochs, self.loss_bbox, label='Loss_bbox', color='red')
            self.ax.set_title('Training Loss Over Epochs')
            self.ax.set_xlabel('Epoch')
            self.ax.set_ylabel('Value')
            self.ax.legend()
            self.ax.grid(True)
            plt.draw()
            plt.pause(0.1)

    def after_run(self, runner):
        # 訓練結束後更新圖表並保存
        self.ax.clear()
        self.ax.plot(self.epochs, self.loss, label='Loss', color='blue')
        self.ax.plot(self.epochs, self.loss_cls, label='Loss_cls', color='green')
        self.ax.plot(self.epochs, self.loss_bbox, label='Loss_bbox', color='red')
        self.ax.set_title('Training Loss Over Epochs')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        self.ax.grid(True)
        plt.draw()
        plt.pause(0.1)
        plt.savefig(self.save_path)
        plt.ioff()
        plt.show()
