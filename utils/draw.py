import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os



class Attention:
    def __init__(self, cmap, attn=None, path=None):
        self.cmap = cmap
        self.attn = attn
        self.path = path

# ...existing code...
    def save_heatmap(self, path: str = None, attn: np.ndarray = None, w: float = None, h: float = None, cmap: str = None, dpi: int = 300):
        """
        支持两种输入：
          - 单张热图：arr 为 2-D (C, H)，行为原来行为；
          - 多张热图：arr 为 3-D 且第0维为 9，形状 (9, C, L)，在一张 3x3 画布上绘制 9 张热力图（共享 vmin/vmax）。
        """
        arr = self.attn if attn is None else attn
        if arr is None:
            raise ValueError('No attention array provided')

        arr = np.asarray(arr)
        save_path = self.path if path is None else path
        if save_path is None:
            raise ValueError('No path provided to save the heatmap')

        cmap = self.cmap if cmap is None else cmap

        # 单张热图（向后兼容）
        if arr.ndim == 2:
            fig_width = w # 109对应宽度为10.9英寸
            fig_height = h # 264对应高度为26.4英寸
            fig = plt.figure(figsize=(fig_width, fig_height))
            plt.imshow(arr, aspect='auto')
            # plt.colorbar() # 添加颜色条以作参考
            plt.show()
            # C, H = arr.shape
            # cell_size = 0.4
            # # minimal sizes based on data
            # fig_w_min = max(3.0, H * cell_size)
            # fig_h_min = max(3.0, C * cell_size)

            # # enforce target aspect ratio width:height = 192:109
            # ratio = 192.0 / 109.0
            # fig_w = max(fig_w_min, fig_h_min * ratio)
            # fig_h = fig_w / ratio

            # # Guard against excessively large pixel dimensions
            # max_pix = 2 ** 16 - 1
            # pix_w = fig_w * dpi
            # pix_h = fig_h * dpi
            # if pix_w > max_pix or pix_h > max_pix:
            #     scale = min(max_pix / pix_w, max_pix / pix_h)
            #     scale = min(scale, 1.0)
            #     fig_w *= scale
            #     fig_h *= scale

            # fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            # im = ax.imshow(arr, origin='lower', interpolation='nearest', aspect='auto', cmap=cmap)
            # plt.xticks([0, 108], [1, 109])
            plt.yticks([0, 167], [1, 168])
            # # x label should be 'Channel', y label 'Inter-cycle'
            plt.xlabel('Inter-cycle')
            plt.ylabel('Time')
            # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            return

        # 多张热图：要求形状为 (9, C, L)
        if arr.ndim != 3 or arr.shape[0] != 9:
            raise ValueError(f'attn must be 2-D (C,H) or 3-D with first dim 9 (9,C,L), got shape {arr.shape}')

        n, C, L = arr.shape
        # 每个子图单元的建议尺寸
        cell_w = max(2.0, L * 0.12)   # 根据列数调整单元格宽度
        cell_h = max(2.0, C * 0.12)   # 根据行数调整单元格高度
        fig_w_min = cell_w * 3 + 0.6      # 多留一点宽度给 colorbar 列
        fig_h_min = cell_h * 3

        # enforce target aspect ratio width:height = 192:109
        ratio = 192.0 / 109.0
        fig_w = max(fig_w_min, fig_h_min * ratio)
        fig_h = fig_w / ratio

        # Guard against excessively large pixel dimensions
        max_pix = 2 ** 16 - 1
        pix_w = fig_w * dpi
        pix_h = fig_h * dpi
        if pix_w > max_pix or pix_h > max_pix:
            scale = min(max_pix / pix_w, max_pix / pix_h)
            scale = min(scale, 1.0)
            fig_w *= scale
            fig_h *= scale

        # 共享 vmin/vmax，保证颜色一致
        vmin = float(arr.min())
        vmax = float(arr.max())

        # 使用 GridSpec：3x4 网格，最后一列专门给 colorbar
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1, 0.08], wspace=0.05, hspace=0.05)

        axes = []
        im = None
        for i in range(9):
            r = i // 3
            c = i % 3
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(arr[i], origin='lower', interpolation='nearest', aspect='auto',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            # x label should be 'Channel', y label 'Inter-cycle'
            ax.set_xlabel('Channel')
            ax.set_ylabel('Inter-cycle')
            axes.append(ax)

        # 在右侧单独列创建颜色条轴（跨三行）
        cax = fig.add_subplot(gs[:, 3])
        # 使用 colorbar 时不要关闭该轴
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        # 根据需要可以微调 colorbar 标签样式
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')

        # 紧凑布局，但避免 colorbar 被压缩
        plt.tight_layout()
        # 保存并关闭
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def save_lineplots(self, arr: np.ndarray, out_dir: str = '.', prefix: str = '', dpi: int = 300):
            """
            保存折线图，支持两种输入形状：
                1) 一维数组，长度 L -> 保存为 x.png（或 prefix+x.png）
                2) 二维数组，形状 (C, L) -> 为每个通道 c 保存一张图，命名为 {c}.png（或 prefix+{c}.png），共生成 C 个文件

            图像要求：仅在 x 轴标注 `'time'`；画布宽高比为 3:1（width:height = 3:1）。

            参数：
                - arr: numpy 数组，1-D 或 2-D
                - out_dir: 保存目录（若不存在则尝试创建）
                - prefix: 文件名前缀（默认为空）
                - dpi: 保存分辨率
            """

            L = arr.shape[1]
            fig_w = 6.0  # width
            fig_h = 6.0  # height (ratio 3:1)
            fig, ax = plt.subplots(3, 1, figsize=(fig_w, fig_h))
            x = np.arange(L)

            ax[0].plot(x, arr[0], '-', color='blue', label=r'$\mathbf{X}_{local}^{}$')
            ax[0].legend(loc='upper center', fontsize=12)
            # ax[0].set_xlabel('time')
            # ax[0].set_ylabel('value')
            ax[0].set_xlim(x[0], x[-1])

            ax[1].plot(x, arr[1], '-', color='red', label=r'$\mathbf{X}_{global}^{}$')
            ax[1].legend(loc='upper center', fontsize=12)
            # ax[0].set_xlabel('time')
            ax[1].set_ylabel('value')
            ax[1].set_xlim(x[0], x[-1])

            ax[2].plot(x, arr[2], '-', color='green', label=r'$\mathbf{X}_{enc}^{}$')
            ax[2].legend(loc='upper center', fontsize=12)
            ax[2].set_xlabel('time')
            # ax[2].set_ylabel('value')
            ax[2].set_xlim(x[0], x[-1])

            plt.tight_layout()
            plt.savefig(out_dir, dpi=dpi, bbox_inches='tight')

# ...existing code...