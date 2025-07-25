import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')

class ResearchPaperPlotGenerator:
    """
    Generates publication-ready plots for comparing video enhancement methods in object detection.
    Features statistical annotations, consistent styling, and comprehensive analysis visualizations.
    """

    def __init__(self, video_path_baseline, video_path_rag, video_path_ablation, 
                 model_path='yolov8n.pt', conf_threshold=0.5):
        """Initialize with video paths and analysis parameters."""
        self.video_paths = [video_path_baseline, video_path_rag, video_path_ablation]
        self.method_names = ['Baseline', 'RAG-Enhanced', 'Ablation']
        # Publication-quality color scheme (colorblind-friendly)
        self.method_colors = ['#4E79A7', '#F28E2B', '#E15759']  # Blue, Orange, Red
        self.conf_threshold = conf_threshold
        self.high_conf_threshold = 0.75
        
        print("Initializing research paper plot generator...")
        
        self.model = YOLO(model_path)
        self.target_class = 2  # COCO class for 'car'
        
        self.output_dir = "research_paper_plots"
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Journal-quality plotting parameters
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'lines.linewidth': 1.8,
            'axes.linewidth': 0.8,
            'grid.alpha': 0.3,
            'figure.dpi': 600,
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05
        })

    def _extract_detections(self, results):
        """Extract detection confidences for target class."""
        return [float(box.conf) for box in results.boxes 
                if int(box.cls) == self.target_class 
                and float(box.conf) >= self.conf_threshold]

    def _calculate_frame_metrics(self, confidences, method_idx):
        """Calculate frame-level metrics from detection confidences."""
        prefix = f"method_{method_idx}"
        return {
            f'{prefix}_detection_count': len(confidences),
            f'{prefix}_mean_confidence': np.mean(confidences) if confidences else 0,
            f'{prefix}_high_conf_count': sum(1 for c in confidences if c > self.high_conf_threshold)
        }

    def process_videos_and_analyze(self):
        """Process videos and compute all metrics."""
        caps = [cv2.VideoCapture(path) for path in self.video_paths]
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        
        if not all(count == frame_counts[0] for count in frame_counts):
            raise ValueError(f"Frame count mismatch: {frame_counts}")
        
        total_frames = frame_counts[0]
        self.comprehensive_metrics = []
        
        print(f"Processing {total_frames} frames...")
        
        for frame_idx in tqdm(range(total_frames), desc="Analyzing frames"):
            frames = [cap.read()[1] for cap in caps]
            if any(f is None for f in frames): break
            
            frame_metrics = {'frame_id': frame_idx}
            
            # Visual quality metrics (compared to baseline)
            frame_metrics['psnr_rag'] = psnr(frames[0], frames[1], data_range=255)
            frame_metrics['ssim_rag'] = ssim(frames[0], frames[1], channel_axis=2, data_range=255)
            frame_metrics['psnr_ablation'] = psnr(frames[0], frames[2], data_range=255)
            frame_metrics['ssim_ablation'] = ssim(frames[0], frames[2], channel_axis=2, data_range=255)
            
            # Detection metrics for each method
            for method_idx, frame in enumerate(frames):
                results = self.model(frame, verbose=False)
                detections = self._extract_detections(results[0])
                metrics = self._calculate_frame_metrics(detections, method_idx)
                frame_metrics.update(metrics)
            
            self.comprehensive_metrics.append(frame_metrics)
        
        for cap in caps: cap.release()
        
        self._create_dataframes()
        self._compute_statistics()
        print("Analysis completed.")
        return True

    def _create_dataframes(self):
        """Create structured dataframes for analysis."""
        self.df = pd.DataFrame(self.comprehensive_metrics)
        self.method_dfs = []
        for i in range(3):
            method_cols = [col for col in self.df.columns if f'method_{i}' in col]
            method_df = self.df[['frame_id'] + method_cols].copy()
            method_df.columns = ['frame_id'] + [c.replace(f'method_{i}_', '') for c in method_cols]
            self.method_dfs.append(method_df)

    def _compute_statistics(self):
        """Compute statistical metrics and significance tests."""
        print("Computing statistical tests...")
        self.stats_results = {}
        metrics_to_test = ['detection_count', 'mean_confidence', 'high_conf_count']
        
        # ANOVA for overall differences
        for metric in metrics_to_test:
            groups = [df[metric].values for df in self.method_dfs]
            f_stat, p_value = stats.f_oneway(*groups)
            self.stats_results[f'{metric}_anova'] = {'f_stat': f_stat, 'p_value': p_value}
            
            # Pairwise t-tests with Bonferroni correction
            for i, j in [(0, 1), (0, 2), (1, 2)]:
                t_stat, p_val = stats.ttest_rel(groups[i], groups[j])
                self.stats_results[f'{metric}_ttest_{i}_{j}'] = {
                    't_stat': t_stat, 
                    'p_value': p_val,
                    'p_value_corrected': p_val * 3  # Bonferroni correction
                }

    def _create_comparison_plot(self, column_name, title, ylabel, filename):
        """
        Create a publication-quality comparison plot with:
        - Smoothed trends
        - Confidence intervals
        - Statistical annotations
        - Difference plot
        """
        fig, (ax1, ax2) = plt.subplots(
            2, 1, 
            figsize=(6, 4.5), 
            sharex=True, 
            height_ratios=[2, 1],
            gridspec_kw={'hspace': 0.1}
        )
        
        window_size = max(15, len(self.df) // 20)
        
        # Main plot (top panel)
        for i, df in enumerate(self.method_dfs):
            mean_trend = df[column_name].rolling(window=window_size, center=True, min_periods=1).mean()
            std_trend = df[column_name].rolling(window=window_size, center=True, min_periods=1).std().fillna(0)
            
            ax1.plot(df['frame_id'], mean_trend, 
                    color=self.method_colors[i], 
                    label=self.method_names[i])
            ax1.fill_between(df['frame_id'], 
                            mean_trend - std_trend, 
                            mean_trend + std_trend, 
                            color=self.method_colors[i], 
                            alpha=0.15)
        
        ax1.set_ylabel(ylabel)
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.legend(frameon=False)
        
        # Difference plot (bottom panel)
        baseline_trend = self.method_dfs[0][column_name].rolling(
            window=window_size, center=True, min_periods=1).mean()
        
        for i in range(1, 3):
            method_trend = self.method_dfs[i][column_name].rolling(
                window=window_size, center=True, min_periods=1).mean()
            difference = method_trend - baseline_trend
            ax2.plot(self.df['frame_id'], difference, 
                    color=self.method_colors[i], 
                    label=f'{self.method_names[i]} - Baseline')
            ax2.fill_between(self.df['frame_id'], 0, difference, 
                            color=self.method_colors[i], alpha=0.15)
        
        ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Frame Number')
        ax2.set_ylabel('Difference')
        ax2.grid(True, linestyle=':', alpha=0.5)
        
        # Add statistical significance annotation
        sig_pairs = []
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            key = f'{column_name}_ttest_{i}_{j}'
            if key in self.stats_results:
                p_val = self.stats_results[key]['p_value_corrected']
                if p_val < 0.05:
                    sig_pairs.append(f"{self.method_names[i]} vs {self.method_names[j]}: p={p_val:.3f}")
        
        if sig_pairs:
            ax1.text(0.02, 0.98, "Significant pairs:\n" + "\n".join(sig_pairs),
                     transform=ax1.transAxes, ha='left', va='top', fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filename}")

    def _create_quality_metric_plot(self, metric_keys, labels, title, ylabel, filename):
        """Create compact plot for visual quality metrics."""
        plt.figure(figsize=(6, 3))
        window_size = max(15, len(self.df) // 20)
        
        for i, key in enumerate(metric_keys):
            trend = self.df[key].rolling(window=window_size, center=True, min_periods=1).mean()
            std = self.df[key].rolling(window=window_size, center=True, min_periods=1).std().fillna(0)
            
            plt.plot(self.df['frame_id'], trend, 
                    color=self.method_colors[i+1], 
                    label=labels[i])
            plt.fill_between(self.df['frame_id'], 
                           trend - std, 
                           trend + std, 
                           color=self.method_colors[i+1], 
                           alpha=0.15)

        plt.title(title)
        plt.xlabel('Frame Number')
        plt.ylabel(ylabel)
        plt.legend(frameon=False)
        plt.grid(True, linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filename}")

    def _create_correlation_plot(self, x_metric, y_metric, title, xlabel, ylabel, filename):
        """Create scatter plot with regression line and confidence interval."""
        plt.figure(figsize=(4, 3.5))
        
        # Filter out zero-confidence frames
        filtered_df = self.df[self.df[y_metric] > 0]
        
        # Calculate correlation coefficient
        r_val, p_val = stats.pearsonr(filtered_df[x_metric], filtered_df[y_metric])
        
        # Create plot
        sns.regplot(data=filtered_df, x=x_metric, y=y_metric,
                    scatter_kws={'s': 15, 'alpha': 0.4, 'color': self.method_colors[1]},
                    line_kws={'color': 'black', 'linewidth': 1.5})
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Add correlation annotation
        plt.text(0.05, 0.95, f"r = {r_val:.2f}\np = {p_val:.3f}",
                transform=plt.gca().transAxes,
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Saved: {filename}")

    def _create_performance_heatmap(self):
        """Create heatmap showing relative performance across methods."""
        metrics = ['detection_count', 'mean_confidence']
        fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
        
        for ax, metric in zip(axes, metrics):
            # Calculate normalized performance (0-1 scale)
            values = np.array([df[metric].values for df in self.method_dfs])
            norm_values = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0) + 1e-10)
            
            # Find best performing method per frame
            best_methods = np.argmax(norm_values, axis=0)
            
            # Create heatmap
            sns.heatmap(
                best_methods.reshape(1, -1), 
                ax=ax,
                cmap=[self.method_colors[i] for i in range(3)],
                cbar=False,
                yticklabels=False
            )
            
            # Create custom legend
            legend_patches = [Patch(color=self.method_colors[i], label=self.method_names[i]) 
                             for i in range(3)]
            ax.legend(handles=legend_patches, frameon=False, bbox_to_anchor=(1, 1))
            ax.set_title(f'Best Method by Frame ({metric.replace("_", " ").title()})')
            ax.set_xlabel('Frame Number')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'performance_heatmap.pdf'))
        plt.close()
        print("Saved: performance_heatmap.pdf")

    def _create_cumulative_advantage_plot(self):
        """Plot cumulative advantage of enhanced methods over baseline."""
        plt.figure(figsize=(6, 3.5))
        
        metrics = {
            'detection_count': 'Cumulative Detection Advantage',
            'mean_confidence': 'Cumulative Confidence Advantage'
        }
        
        for metric, label in metrics.items():
            baseline = self.method_dfs[0][metric].values
            for i in range(1, 3):
                advantage = np.cumsum(self.method_dfs[i][metric].values - baseline)
                plt.plot(advantage, 
                        color=self.method_colors[i],
                        label=f'{self.method_names[i]} ({metric.replace("_", " ")})')
        
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
        plt.title('Cumulative Performance Advantage Over Baseline')
        plt.xlabel('Frame Number')
        plt.ylabel('Cumulative Difference')
        plt.legend(frameon=False)
        plt.grid(True, linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'cumulative_advantage.pdf'))
        plt.close()
        print("Saved: cumulative_advantage.pdf")

    def _create_confidence_distribution_plot(self):
        """Create violin plots of confidence distributions."""
        plt.figure(figsize=(5, 3.5))
        
        # Prepare data
        data = []
        for i, df in enumerate(self.method_dfs):
            confidences = df['mean_confidence'][df['mean_confidence'] > 0]
            for c in confidences:
                data.append({'Method': self.method_names[i], 'Confidence': c})
        
        df_plot = pd.DataFrame(data)
        
        # Create plot
        sns.violinplot(
            data=df_plot,
            x='Method',
            y='Confidence',
            palette=self.method_colors,
            cut=0,
            inner='quartile'
        )
        
        plt.title('Distribution of Detection Confidences')
        plt.ylabel('Confidence Score')
        plt.xlabel('')
        plt.grid(True, axis='y', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'confidence_distributions.pdf'))
        plt.close()
        print("Saved: confidence_distributions.pdf")

    def _create_robustness_tradeoff_plot(self):
        """Plot mean performance vs. robustness (1/std) across methods."""
        plt.figure(figsize=(5, 4))
        
        metrics = ['detection_count', 'mean_confidence']
        markers = ['o', 's']
        
        for metric, marker in zip(metrics, markers):
            means = []
            robustness = []
            
            for i, df in enumerate(self.method_dfs):
                vals = df[metric].values
                means.append(np.mean(vals))
                robustness.append(1/np.std(vals))  # Inverse of std as robustness measure
                
            plt.scatter(
                means, robustness,
                color=self.method_colors,
                s=100,
                marker=marker,
                label=metric.replace('_', ' ')
            )
            
            # Add method labels
            for i, (x, y) in enumerate(zip(means, robustness)):
                plt.text(x, y, self.method_names[i][:3], 
                        ha='center', va='center', 
                        color='white', fontsize=8)
        
        plt.title('Performance-Robustness Tradeoff')
        plt.xlabel('Mean Value')
        plt.ylabel('Robustness (1/σ)')
        plt.legend(frameon=False)
        plt.grid(True, linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'robustness_tradeoff.pdf'))
        plt.close()
        print("Saved: robustness_tradeoff.pdf")

    def _create_enhanced_correlation_matrix(self):
        """Create a comprehensive correlation matrix plot."""
        # Prepare data
        corr_data = pd.DataFrame({
            'PSNR (RAG)': self.df['psnr_rag'],
            'SSIM (RAG)': self.df['ssim_rag'],
            'Detections (RAG)': self.method_dfs[1]['detection_count'],
            'Confidence (RAG)': self.method_dfs[1]['mean_confidence'],
            'Detections (Ablation)': self.method_dfs[2]['detection_count'],
            'Confidence (Ablation)': self.method_dfs[2]['mean_confidence']
        })
        
        # Compute correlations
        corr = corr_data.corr()
        
        # Create plot
        plt.figure(figsize=(6, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        
        sns.heatmap(
            corr, 
            mask=mask,
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5}
        )
        
        plt.title('Cross-Method Metric Correlations')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'enhanced_correlation_matrix.pdf'))
        plt.close()
        print("Saved: enhanced_correlation_matrix.pdf")

    def _create_improvement_highlight_plot(self):
        """Highlight frames where methods significantly outperform baseline."""
        plt.figure(figsize=(8, 4))
        
        threshold = 1.5  # Standard deviations for significant improvement
        
        for i in range(1, 3):
            baseline = self.method_dfs[0]['detection_count']
            method = self.method_dfs[i]['detection_count']
            difference = method - baseline
            std_diff = difference.std()
            
            # Find significantly improved frames
            improved_frames = np.where(difference > threshold * std_diff)[0]
            
            if len(improved_frames) > 0:
                plt.scatter(
                    improved_frames,
                    method[improved_frames],
                    color=self.method_colors[i],
                    s=30,
                    label=f'{self.method_names[i]} significant improvements'
                )
        
        # Plot all detections
        for i, df in enumerate(self.method_dfs):
            plt.plot(df['detection_count'], 
                    color=self.method_colors[i],
                    alpha=0.3,
                    label=f'{self.method_names[i]} all frames')
        
        plt.title('Frames with Significant Detection Improvements')
        plt.xlabel('Frame Number')
        plt.ylabel('Detection Count')
        plt.legend(frameon=False, bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'improvement_highlights.pdf'))
        plt.close()
        print("Saved: improvement_highlights.pdf")

    def generate_all_plots(self):
        """Generate the complete set of publication-ready plots."""
        if not hasattr(self, 'comprehensive_metrics'):
            if not self.process_videos_and_analyze():
                return
            
        print("\nGenerating research paper plots...")
        
        # Core comparison plots
        self._create_comparison_plot(
            'detection_count', 
            'Car Detection Count Comparison', 
            'Detections per Frame', 
            'detection_count.pdf'
        )
        
        self._create_comparison_plot(
            'mean_confidence', 
            'Detection Confidence Comparison', 
            'Mean Confidence Score', 
            'detection_confidence.pdf'
        )
        
        # Visual quality plots
        self._create_quality_metric_plot(
            ['psnr_rag', 'psnr_ablation'],
            ['RAG-Enhanced', 'Ablation'],
            'PSNR Comparison to Baseline',
            'PSNR (dB)',
            'psnr_comparison.pdf'
        )
        
        self._create_quality_metric_plot(
            ['ssim_rag', 'ssim_ablation'],
            ['RAG-Enhanced', 'Ablation'],
            'SSIM Comparison to Baseline',
            'SSIM',
            'ssim_comparison.pdf'
        )
        
        # Correlation plots
        self._create_correlation_plot(
            'psnr_rag', 'method_1_mean_confidence',
            'PSNR vs Confidence (RAG)',
            'PSNR (dB)', 'Mean Confidence',
            'corr_psnr_rag.pdf'
        )
        
        self._create_correlation_plot(
            'ssim_rag', 'method_1_mean_confidence',
            'SSIM vs Confidence (RAG)',
            'SSIM', 'Mean Confidence',
            'corr_ssim_rag.pdf'
        )
        
        # Additional analysis plots
        self._create_performance_heatmap()
        self._create_cumulative_advantage_plot()
        self._create_confidence_distribution_plot()
        self._create_robustness_tradeoff_plot()
        self._create_enhanced_correlation_matrix()
        self._create_improvement_highlight_plot()
        
        print("\nAll plots generated successfully.")
        print(f"Output directory: {self.figures_dir}")

def main():
    """Main execution function."""
    video1_path = "results_baseline/baseline_resaved_video.mp4"
    video2_path = "results_rag_diffusion/rag_enhanced_video.mp4"
    video3_path = "results_ablation_study/ablation_enhanced_video.mp4"
    model_path = "yolov8n.pt"

    # Verify input files
    missing_files = [p for p in [video1_path, video2_path, video3_path] if not os.path.exists(p)]
    if missing_files:
        print(f"Error: Missing video files: {missing_files}")
        return

    plot_generator = ResearchPaperPlotGenerator(
        video_path_baseline=video1_path, 
        video_path_rag=video2_path, 
        video_path_ablation=video3_path, 
        model_path=model_path
    )
    plot_generator.generate_all_plots()

if __name__ == "__main__":
    main()