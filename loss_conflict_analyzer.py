import torch
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import csv
from datetime import datetime
import warnings


class LossConflictAnalyzer:
    """
    Analyzer to detect and quantify conflicts between different loss functions
    during training by analyzing gradient directions, loss correlations, and
    gradient magnitudes.
    """
    
    def __init__(self, loss_names, model, log_freq=100, buffer_size=500, save_dir='conflict_analysis'):
        """
        Args:
            loss_names: List of loss function names to track
            model: PyTorch model to extract gradients from
            log_freq: Frequency of logging conflict analysis
            buffer_size: Number of samples to keep for correlation analysis
            save_dir: Directory to save analysis results
        """
        self.loss_names = loss_names
        self.model = model
        self.log_freq = log_freq
        self.buffer_size = buffer_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Buffers for tracking
        self.loss_history = {name: deque(maxlen=buffer_size) for name in loss_names}
        self.gradient_cache = {}
        self.step_count = 0
        
        # Statistics storage
        self.conflict_scores = defaultdict(list)
        self.correlation_scores = defaultdict(list)
        self.gradient_magnitudes = defaultdict(list)
        
        # CSV logging
        self.csv_path = self.save_dir / f'conflict_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self._init_csv()
        
        print(f"📊 LossConflictAnalyzer initialized")
        print(f"   Tracking losses: {', '.join(loss_names)}")
        print(f"   Save directory: {self.save_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['iteration', 'loss1', 'loss2', 'gradient_cosine', 
                     'loss_correlation', 'grad_magnitude_ratio']
            writer.writerow(header)
    
    def _get_model_gradients(self):
        """Extract all gradients from model parameters"""
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        
        if len(gradients) == 0:
            return None
        return torch.cat(gradients)
    
    def compute_individual_loss_gradient(self, loss, loss_name):
        """
        Compute gradient for individual loss by performing backward pass
        
        Args:
            loss: Individual loss tensor
            loss_name: Name of the loss
        """
        # Zero gradients first
        self.model.zero_grad()
        
        # Backward pass for this loss only
        if loss.requires_grad:
            loss.backward(retain_graph=True)
            
            # Extract gradients
            grad = self._get_model_gradients()
            
            if grad is not None:
                # Store gradient
                self.gradient_cache[loss_name] = grad.detach().clone()
                
                # Track magnitude
                grad_magnitude = torch.norm(grad).item()
                self.gradient_magnitudes[loss_name].append(grad_magnitude)
        
        # Clear gradients for next loss
        self.model.zero_grad()
    
    def compute_gradient_conflict(self, loss_name1, loss_name2):
        """
        Compute cosine similarity between gradients of two losses
        
        Returns:
            float: Cosine similarity in [-1, 1]
                   1 = same direction (no conflict)
                   0 = orthogonal
                  -1 = opposite direction (high conflict)
        """
        if loss_name1 not in self.gradient_cache or loss_name2 not in self.gradient_cache:
            return None
        
        grad1 = self.gradient_cache[loss_name1]
        grad2 = self.gradient_cache[loss_name2]
        
        # Cosine similarity
        cos_sim = torch.dot(grad1, grad2) / (torch.norm(grad1) * torch.norm(grad2) + 1e-8)
        
        return cos_sim.item()
    
    def compute_loss_correlation(self, loss_name1, loss_name2):
        """
        Compute Pearson correlation between two loss value histories
        
        Returns:
            float: Correlation coefficient in [-1, 1]
        """
        if len(self.loss_history[loss_name1]) < 10:
            return None
        
        values1 = np.array(self.loss_history[loss_name1])
        values2 = np.array(self.loss_history[loss_name2])
        
        # Ensure same length
        min_len = min(len(values1), len(values2))
        values1 = values1[-min_len:]
        values2 = values2[-min_len:]
        
        # Compute correlation
        correlation = np.corrcoef(values1, values2)[0, 1]
        
        return correlation if not np.isnan(correlation) else None
    
    def analyze_step(self, losses_dict, writer=None):
        """
        Analyze conflicts for current training step
        
        Args:
            losses_dict: Dictionary mapping loss names to loss tensors
                        e.g., {'ddpm': ddpm_loss, 'dan_expr': dan_expr_loss, ...}
            writer: Optional TensorBoard SummaryWriter
        
        Returns:
            dict: Conflict analysis results
        """
        self.step_count += 1
        
        # Update loss history
        for name, loss in losses_dict.items():
            if name in self.loss_names:
                loss_value = loss.item() if torch.is_tensor(loss) else loss
                self.loss_history[name].append(loss_value)
        
        # Compute individual gradients
        for name, loss in losses_dict.items():
            if name in self.loss_names and torch.is_tensor(loss):
                self.compute_individual_loss_gradient(loss, name)
        
        # Analyze conflicts every log_freq steps
        if self.step_count % self.log_freq == 0:
            results = self._perform_analysis(writer)
            return results
        
        return None
    
    def _perform_analysis(self, writer=None):
        """Perform comprehensive conflict analysis"""
        results = {
            'gradient_conflicts': {},
            'loss_correlations': {},
            'gradient_magnitudes': {},
            'warnings': []
        }
        
        # Analyze all pairs
        for i, name1 in enumerate(self.loss_names):
            for j, name2 in enumerate(self.loss_names):
                if i >= j:  # Skip diagonal and duplicates
                    continue
                
                pair_key = f"{name1}↔{name2}"
                
                # Gradient conflict
                cos_sim = self.compute_gradient_conflict(name1, name2)
                if cos_sim is not None:
                    results['gradient_conflicts'][pair_key] = cos_sim
                    self.conflict_scores[pair_key].append(cos_sim)
                    
                    # Check for high conflict
                    if cos_sim < -0.3:
                        warning = f"⚠️ HIGH GRADIENT CONFLICT: {pair_key} = {cos_sim:.3f}"
                        results['warnings'].append(warning)
                
                # Loss correlation
                correlation = self.compute_loss_correlation(name1, name2)
                if correlation is not None:
                    results['loss_correlations'][pair_key] = correlation
                    self.correlation_scores[pair_key].append(correlation)
                    
                    # Check for negative correlation
                    if correlation < -0.4:
                        warning = f"⚠️ NEGATIVE LOSS CORRELATION: {pair_key} = {correlation:.3f}"
                        results['warnings'].append(warning)
                
                # Gradient magnitude ratio
                if name1 in self.gradient_magnitudes and name2 in self.gradient_magnitudes:
                    mag1 = self.gradient_magnitudes[name1][-1] if self.gradient_magnitudes[name1] else 0
                    mag2 = self.gradient_magnitudes[name2][-1] if self.gradient_magnitudes[name2] else 0
                    
                    if mag2 > 1e-8:
                        ratio = mag1 / mag2
                        if ratio > 100 or ratio < 0.01:
                            warning = f"⚠️ GRADIENT MAGNITUDE IMBALANCE: {name1}/{name2} = {ratio:.1f}x"
                            results['warnings'].append(warning)
                
                # Log to CSV
                if cos_sim is not None and correlation is not None:
                    mag_ratio = mag1 / (mag2 + 1e-8) if 'mag1' in locals() else 0
                    self._log_to_csv(name1, name2, cos_sim, correlation, mag_ratio)
        
        # Track individual gradient magnitudes
        for name in self.loss_names:
            if name in self.gradient_magnitudes and self.gradient_magnitudes[name]:
                results['gradient_magnitudes'][name] = self.gradient_magnitudes[name][-1]
        
        # Print summary
        self._print_summary(results)
        
        # Log to TensorBoard
        if writer is not None:
            self._log_to_tensorboard(writer, results)
        
        # Visualize periodically
        if self.step_count % (self.log_freq * 10) == 0:
            self.visualize_conflicts()
        
        return results
    
    def _print_summary(self, results):
        """Print conflict analysis summary"""
        print(f"\n{'='*60}")
        print(f"📊 Loss Conflict Analysis (iteration {self.step_count})")
        print(f"{'='*60}")
        
        # Gradient conflicts
        if results['gradient_conflicts']:
            print("\n🔍 Gradient Conflicts (cosine similarity):")
            for pair, score in sorted(results['gradient_conflicts'].items(), key=lambda x: x[1]):
                status = self._get_conflict_status(score)
                print(f"  {pair}: {score:+.3f} {status}")
        
        # Loss correlations
        if results['loss_correlations']:
            print("\n📈 Loss Correlations (last {} samples):".format(self.buffer_size))
            for pair, corr in sorted(results['loss_correlations'].items(), key=lambda x: x[1]):
                status = "⚠️" if corr < -0.4 else "✓" if corr > 0.4 else "○"
                print(f"  {pair}: {corr:+.3f} {status}")
        
        # Gradient magnitudes
        if results['gradient_magnitudes']:
            print("\n⚡ Gradient Magnitudes:")
            for name, mag in sorted(results['gradient_magnitudes'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {mag:.6f}")
        
        # Warnings
        if results['warnings']:
            print("\n⚠️  WARNINGS:")
            for warning in results['warnings']:
                print(f"  {warning}")
        
        print(f"{'='*60}\n")
    
    def _get_conflict_status(self, cos_sim):
        """Get status emoji based on cosine similarity"""
        if cos_sim > 0.7:
            return "✓ (aligned)"
        elif cos_sim > 0.3:
            return "✓ (weak align)"
        elif cos_sim > -0.3:
            return "○ (independent)"
        elif cos_sim > -0.7:
            return "❌ (conflict)"
        else:
            return "❌❌ (strong conflict)"
    
    def _log_to_csv(self, loss1, loss2, cos_sim, correlation, grad_ratio):
        """Log conflict metrics to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step_count,
                loss1,
                loss2,
                f"{cos_sim:.6f}",
                f"{correlation:.6f}",
                f"{grad_ratio:.6f}"
            ])
    
    def _log_to_tensorboard(self, writer, results):
        """Log metrics to TensorBoard"""
        step = self.step_count
        
        # Log gradient conflicts
        for pair, score in results['gradient_conflicts'].items():
            writer.add_scalar(f"Conflict/Gradient/{pair}", score, step)
        
        # Log correlations
        for pair, corr in results['loss_correlations'].items():
            writer.add_scalar(f"Conflict/Correlation/{pair}", corr, step)
        
        # Log gradient magnitudes
        for name, mag in results['gradient_magnitudes'].items():
            writer.add_scalar(f"Conflict/GradMagnitude/{name}", mag, step)
    
    def visualize_conflicts(self):
        """Create heatmap visualization of conflict scores"""
        if not self.conflict_scores:
            print("⚠️ Not enough data for visualization yet")
            return
        
        n_losses = len(self.loss_names)
        
        # Create matrices for visualization
        gradient_conflict_matrix = np.zeros((n_losses, n_losses))
        correlation_matrix = np.zeros((n_losses, n_losses))
        
        # Fill matrices
        for i, name1 in enumerate(self.loss_names):
            for j, name2 in enumerate(self.loss_names):
                if i == j:
                    gradient_conflict_matrix[i, j] = 1.0  # Self-similarity
                    correlation_matrix[i, j] = 1.0
                elif i < j:
                    pair_key = f"{name1}↔{name2}"
                    
                    # Average over recent history
                    if pair_key in self.conflict_scores and self.conflict_scores[pair_key]:
                        grad_conf = np.mean(self.conflict_scores[pair_key][-10:])
                        gradient_conflict_matrix[i, j] = grad_conf
                        gradient_conflict_matrix[j, i] = grad_conf
                    
                    if pair_key in self.correlation_scores and self.correlation_scores[pair_key]:
                        corr = np.mean(self.correlation_scores[pair_key][-10:])
                        correlation_matrix[i, j] = corr
                        correlation_matrix[j, i] = corr
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gradient conflict heatmap
        sns.heatmap(gradient_conflict_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0,
                   vmin=-1,
                   vmax=1,
                   xticklabels=self.loss_names,
                   yticklabels=self.loss_names,
                   ax=axes[0],
                   cbar_kws={'label': 'Cosine Similarity'})
        axes[0].set_title('Gradient Conflict Matrix\n(Red=Conflict, Green=Aligned)', fontsize=12, fontweight='bold')
        
        # Loss correlation heatmap
        sns.heatmap(correlation_matrix,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0,
                   vmin=-1,
                   vmax=1,
                   xticklabels=self.loss_names,
                   yticklabels=self.loss_names,
                   ax=axes[1],
                   cbar_kws={'label': 'Correlation'})
        axes[1].set_title('Loss Correlation Matrix\n(Red=Negative, Green=Positive)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.save_dir / f'conflict_heatmap_iter{self.step_count}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved conflict visualization: {save_path}")
        plt.close()
    
    def get_summary_statistics(self):
        """Get summary statistics of conflicts over entire training"""
        summary = {}
        
        for pair_key in self.conflict_scores.keys():
            grad_conflicts = self.conflict_scores[pair_key]
            correlations = self.correlation_scores.get(pair_key, [])
            
            summary[pair_key] = {
                'gradient_conflict': {
                    'mean': np.mean(grad_conflicts) if grad_conflicts else None,
                    'std': np.std(grad_conflicts) if grad_conflicts else None,
                    'min': np.min(grad_conflicts) if grad_conflicts else None,
                    'max': np.max(grad_conflicts) if grad_conflicts else None,
                },
                'loss_correlation': {
                    'mean': np.mean(correlations) if correlations else None,
                    'std': np.std(correlations) if correlations else None,
                }
            }
        
        return summary
    
    def save_final_report(self):
        """Generate final comprehensive report"""
        summary = self.get_summary_statistics()
        
        report_path = self.save_dir / 'final_conflict_report.txt'
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LOSS CONFLICT ANALYSIS - FINAL REPORT\n")
            f.write(f"Total iterations analyzed: {self.step_count}\n")
            f.write("="*80 + "\n\n")
            
            for pair_key, stats in summary.items():
                f.write(f"\n{pair_key}\n")
                f.write("-" * 40 + "\n")
                
                grad_stats = stats['gradient_conflict']
                f.write(f"Gradient Conflict (cosine similarity):\n")
                f.write(f"  Mean: {grad_stats['mean']:.4f}\n")
                f.write(f"  Std:  {grad_stats['std']:.4f}\n")
                f.write(f"  Range: [{grad_stats['min']:.4f}, {grad_stats['max']:.4f}]\n")
                
                corr_stats = stats['loss_correlation']
                if corr_stats['mean'] is not None:
                    f.write(f"\nLoss Correlation:\n")
                    f.write(f"  Mean: {corr_stats['mean']:.4f}\n")
                    f.write(f"  Std:  {corr_stats['std']:.4f}\n")
                
                f.write("\n")
        
        print(f"📄 Saved final report: {report_path}")


# Standalone testing function
if __name__ == "__main__":
    print("LossConflictAnalyzer module loaded successfully")
    print("\nUsage example:")
    print("""
    from loss_conflict_analyzer import LossConflictAnalyzer
    
    # Initialize
    analyzer = LossConflictAnalyzer(
        loss_names=['ddpm', 'dan_expr', 'wav', 'id', 'lpips', 'cycle'],
        model=model,
        log_freq=100
    )
    
    # In training loop (before optimizer.step())
    conflict_results = analyzer.analyze_step({
        'ddpm': ddpm_loss,
        'dan_expr': dan_expr_loss,
        'wav': wav_loss,
        'id': id_loss,
        'lpips': lpips_loss,
        'cycle': cycle_loss
    }, writer=tensorboard_writer)
    
    # At end of training
    analyzer.save_final_report()
    """)
