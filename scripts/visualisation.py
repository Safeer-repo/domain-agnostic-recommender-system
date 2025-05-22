import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import os

# Define the path to the Downloads folder
downloads_folder = os.path.expanduser("~/Downloads")

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Sample data based on your report (replace with actual metrics)
# E-commerce domain metrics
ecommerce_metrics = pd.DataFrame({
    'Algorithm': ['ALS', 'Baseline', 'SAR', 'SVD', 'LightFM'],
    'Precision@K': [0.00067, 0.00100, 0.00001, 0.00010, 0.00010],
    'Recall@K': [0.00418, 0.01000, 0.00007, 0.00053, 0.00100],
    'NDCG@K': [0.00151, 0.00333, 0.00005, 0.00030, 0.00030],
    'MAP@K': [0.08056, 0.00143, 0.00004, 0.00016, 0.11111],
    'Coverage': [0.00559, 0.00011, 0.15044, 0.03610, 0.00123],
    'Novelty': [11.2942, 9.20679, 18.0894, 15.1626, 18.7105],
    'Diversity': [0.00118, 1.00000, 1.00000, 1.00000, 1.00000],
    'Training_Time': [157.09, 6.07, 10.85, 1188.51, 659.18]
})

# Entertainment domain metrics
entertainment_metrics = pd.DataFrame({
    'Algorithm': ['ALS', 'Baseline', 'SAR', 'SVD', 'LightFM'],
    'Precision@K': [0.10636, 0.32500, 0.01091, 0.18882, np.nan],
    'Recall@K': [0.05566, 0.05175, 0.01053, 0.03960, np.nan],
    'NDCG@K': [0.12138, 0.33074, 0.01300, 0.16981, np.nan],
    'MAP@K': [0.15339, 0.21978, 0.00593, 0.08681, np.nan],
    'Coverage': [0.42023, 0.00872, 0.05929, 0.13078, np.nan],
    'Novelty': [9.28510, 7.64502, 13.34976, 9.05208, np.nan],
    'Diversity': [1.00000, 1.00000, 1.00000, 1.00000, np.nan],
    'Training_Time': [0.35, 0.01, 0.11, 12.87, 4.74]
})

# Function to normalize metrics for visualization
def normalize_metrics(df, metrics_to_normalize):
    df_norm = df.copy()
    for metric in metrics_to_normalize:
        if metric == 'Training_Time':
            # For training time, lower is better, so invert the normalization
            max_val = df[metric].max()
            df_norm[metric] = 1 - (df[metric] / max_val)
        else:
            # For other metrics, higher is better
            max_val = df[metric].max()
            if max_val > 0:  # Avoid division by zero
                df_norm[metric] = df[metric] / max_val
    return df_norm

# Normalize metrics for radar charts
metrics_to_normalize = ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K', 
                       'Coverage', 'Novelty', 'Diversity', 'Training_Time']

ecommerce_norm = normalize_metrics(ecommerce_metrics, metrics_to_normalize)
entertainment_norm = normalize_metrics(entertainment_metrics, metrics_to_normalize)

# 1. RADAR CHART: Algorithm Comparison
def create_algorithm_comparison_radar(domain_metrics, domain_name):
    metrics = ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP@K', 
              'Coverage', 'Novelty', 'Diversity']
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add a line for each algorithm
    for i, alg in enumerate(domain_metrics['Algorithm'].unique()):
        values = domain_metrics[domain_metrics['Algorithm'] == alg][metrics].iloc[0].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot the line
        ax.plot(angles, values, linewidth=2, label=alg)
        # Fill area
        ax.fill(angles, values, alpha=0.1)
    
    # Add metrics labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Draw y-axis labels (0-1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add title and legend
    plt.title(f'Algorithm Performance Comparison - {domain_name} Domain', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

# 2. BAR CHART: Cross-Domain Performance
def create_cross_domain_performance_chart(metric='MAP@K'):
    # Prepare data
    data = pd.DataFrame({
        'Algorithm': ecommerce_metrics['Algorithm'],
        'E-commerce': ecommerce_metrics[metric],
        'Entertainment': entertainment_metrics[metric]
    })
    
    # Melt the DataFrame for easier plotting
    melted_data = pd.melt(data, id_vars=['Algorithm'], 
                         value_vars=['E-commerce', 'Entertainment'],
                         var_name='Domain', value_name='Value')
    
    # Create plot
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(x='Algorithm', y='Value', hue='Domain', data=melted_data)
    
    # Add value labels on bars
    for p in chart.patches:
        height = p.get_height()
        if not np.isnan(height):
            chart.text(p.get_x() + p.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha="center", fontsize=9)
    
    plt.title(f'Cross-Domain {metric} Comparison', fontsize=15)
    plt.xlabel('Algorithm')
    plt.ylabel(metric)
    plt.tight_layout()
    
    return plt.gcf()

# 3. SCATTER PLOT: Training Time vs. Performance
def create_training_time_performance_plot(domain_metrics, domain_name, perf_metric='MAP@K'):
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot where size corresponds to coverage
    scatter = plt.scatter(domain_metrics['Training_Time'], 
                        domain_metrics[perf_metric],
                        s=domain_metrics['Coverage']*1000,  # Scale coverage for visibility
                        alpha=0.6)
    
    # Add labels
    for i, alg in enumerate(domain_metrics['Algorithm']):
        plt.annotate(alg, 
                   (domain_metrics['Training_Time'].iloc[i], domain_metrics[perf_metric].iloc[i]),
                   xytext=(7, 0), 
                   textcoords='offset points')
    
    plt.xscale('log')  # Log scale for training time
    plt.xlabel('Training Time (s) - Log Scale')
    plt.ylabel(perf_metric)
    plt.title(f'Training Time vs. {perf_metric} - {domain_name} Domain', fontsize=15)
    plt.grid(True, alpha=0.3)
    
    # Add a colorbar legend
    plt.colorbar(scatter, label='Coverage')
    
    return plt.gcf()

# 4. HEATMAP: Hyperparameter Sensitivity (example for ALS)
def create_hyperparameter_sensitivity_heatmap():
    # Sample hyperparameter sensitivity data (replace with actual data)
    # This represents MAP@K for different combinations of factors and regularization
    factors = [50, 100, 150, 200]
    regularization = [0.001, 0.01, 0.1, 1.0]
    
    # Sample performance data for different hyperparameter combinations
    performance_data = np.array([
        [0.071, 0.082, 0.065, 0.042],
        [0.080, 0.095, 0.078, 0.055],
        [0.076, 0.090, 0.072, 0.048],
        [0.068, 0.081, 0.061, 0.039]
    ])
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(performance_data, annot=True, fmt='.3f', 
                   xticklabels=regularization, yticklabels=factors, 
                   cmap='viridis')
    
    plt.xlabel('Regularization')
    plt.ylabel('Factors')
    plt.title('ALS Hyperparameter Sensitivity - MAP@K', fontsize=15)
    
    return plt.gcf()

# 5. VIOLIN PLOT: Performance Distribution
def create_performance_distribution_violin():
    # For this plot, you'd need per-user metrics, which aren't in the summary data
    # This is a placeholder with simulated data
    
    # Create simulated per-user MAP@K data
    np.random.seed(42)
    user_data = pd.DataFrame()
    algorithms = ['ALS', 'Baseline', 'SAR', 'SVD', 'LightFM']
    
    for alg in algorithms:
        # Simulate different distributions for each algorithm
        if alg == 'ALS':
            values = np.random.beta(2, 5, 100) * 0.2
        elif alg == 'Baseline':
            values = np.random.beta(1.5, 5, 100) * 0.3
        elif alg == 'SAR':
            values = np.random.beta(1, 10, 100) * 0.1
        elif alg == 'SVD':
            values = np.random.beta(1.7, 7, 100) * 0.15
        else: # LightFM
            values = np.random.beta(2.5, 4, 100) * 0.25
        
        temp_df = pd.DataFrame({
            'Algorithm': alg,
            'MAP@K': values
        })
        user_data = pd.concat([user_data, temp_df])
    
    # Create violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Algorithm', y='MAP@K', data=user_data, palette='Set3', inner='quartile')
    
    plt.title('Distribution of MAP@K Across Users', fontsize=15)
    plt.xlabel('Algorithm')
    plt.ylabel('MAP@K')
    
    return plt.gcf()

# 6. LINE CHART: Cold Start Performance
def create_cold_start_performance_chart():
    # Simulated data for cold start performance improvement
    ratings_count = [1, 2, 3, 5, 10, 15, 20, 30]
    
    # MAP@K values for different algorithms as rating count increases
    als_perf = [0.01, 0.03, 0.05, 0.08, 0.12, 0.14, 0.16, 0.17]
    baseline_perf = [0.10, 0.11, 0.12, 0.13, 0.14, 0.14, 0.15, 0.15]
    sar_perf = [0.001, 0.002, 0.003, 0.005, 0.007, 0.009, 0.010, 0.011]
    svd_perf = [0.005, 0.01, 0.02, 0.04, 0.07, 0.09, 0.10, 0.11]
    lightfm_perf = [0.02, 0.05, 0.08, 0.12, 0.15, 0.17, 0.18, 0.19]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ratings_count, als_perf, marker='o', label='ALS')
    plt.plot(ratings_count, baseline_perf, marker='s', label='Baseline')
    plt.plot(ratings_count, sar_perf, marker='^', label='SAR')
    plt.plot(ratings_count, svd_perf, marker='d', label='SVD')
    plt.plot(ratings_count, lightfm_perf, marker='*', label='LightFM')
    
    plt.xlabel('Number of User Ratings')
    plt.ylabel('MAP@K')
    plt.title('Cold Start Performance: MAP@K vs. Number of Ratings', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

# 7. SPIDER CHART: Beyond-Accuracy Metrics
def create_beyond_accuracy_spider_chart(domain_metrics, domain_name):
    metrics = ['Coverage', 'Novelty', 'Diversity']
    
    # Normalize novelty for the spider chart (0-1 scale)
    domain_metrics_normalized = domain_metrics.copy()
    max_novelty = domain_metrics['Novelty'].max()
    domain_metrics_normalized['Novelty'] = domain_metrics_normalized['Novelty'] / max_novelty
    
    # Number of variables
    N = len(metrics)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add a line for each algorithm
    for i, alg in enumerate(domain_metrics_normalized['Algorithm'].unique()):
        values = domain_metrics_normalized[domain_metrics_normalized['Algorithm'] == alg][metrics].iloc[0].values.tolist()
        values += values[:1]  # Close the loop
        
        # Plot the line
        ax.plot(angles, values, linewidth=2, label=alg)
        # Fill area
        ax.fill(angles, values, alpha=0.1)
    
    # Add metrics labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Draw y-axis labels (0-1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add title and legend
    plt.title(f'Beyond-Accuracy Metrics Comparison - {domain_name} Domain', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

# 8. BUBBLE CHART: Multi-Dimensional Performance Visualization with Plotly
def create_bubble_chart_performance(domain_metrics, domain_name):
    # Make a copy of the dataframe and drop rows with NaN values
    filtered_metrics = domain_metrics.dropna(subset=['Precision@K', 'Recall@K', 'Coverage']).copy()
    
    # Check if we have enough data to create the chart
    if len(filtered_metrics) < 1:
        # Create a simple message instead of the bubble chart
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Insufficient data for {domain_name} domain bubble chart", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        return plt.gcf()
        
    # Create the bubble chart with the filtered data
    fig = px.scatter(filtered_metrics, 
                   x='Precision@K', y='Recall@K',
                   size='Coverage', color='Algorithm',
                   hover_name='Algorithm',
                   text='Algorithm',
                   size_max=60,
                   title=f'Multi-Dimensional Algorithm Performance - {domain_name} Domain')
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        height=600, width=900,
        xaxis_title='Precision@K',
        yaxis_title='Recall@K'
    )
    
    return fig

# Generate all visualizations
# 1. Radar charts for algorithm comparison
radar_ecommerce = create_algorithm_comparison_radar(ecommerce_norm, 'E-commerce')
radar_entertainment = create_algorithm_comparison_radar(entertainment_norm, 'Entertainment')

# 2. Bar chart for cross-domain performance
bar_map = create_cross_domain_performance_chart('MAP@K')
bar_precision = create_cross_domain_performance_chart('Precision@K')

# 3. Scatter plots for training time vs performance
scatter_ecommerce = create_training_time_performance_plot(ecommerce_metrics, 'E-commerce')
scatter_entertainment = create_training_time_performance_plot(entertainment_metrics, 'Entertainment')

# 4. Heatmap for hyperparameter sensitivity
heatmap = create_hyperparameter_sensitivity_heatmap()

# 5. Violin plot for performance distribution
violin = create_performance_distribution_violin()

# 6. Line chart for cold start performance
cold_start = create_cold_start_performance_chart()

# 7. Spider charts for beyond-accuracy metrics
spider_ecommerce = create_beyond_accuracy_spider_chart(ecommerce_metrics, 'E-commerce')
spider_entertainment = create_beyond_accuracy_spider_chart(entertainment_metrics, 'Entertainment')

# 8. Bubble charts with plotly for multi-dimensional visualization
bubble_ecommerce = create_bubble_chart_performance(ecommerce_metrics, 'E-commerce')
bubble_entertainment = create_bubble_chart_performance(entertainment_metrics, 'Entertainment')

# Save all figures to the Downloads folder
radar_ecommerce.savefig(os.path.join(downloads_folder, 'radar_ecommerce.png'), dpi=300, bbox_inches='tight')
radar_entertainment.savefig(os.path.join(downloads_folder, 'radar_entertainment.png'), dpi=300, bbox_inches='tight')
bar_map.savefig(os.path.join(downloads_folder, 'bar_map.png'), dpi=300, bbox_inches='tight')
bar_precision.savefig(os.path.join(downloads_folder, 'bar_precision.png'), dpi=300, bbox_inches='tight')
scatter_ecommerce.savefig(os.path.join(downloads_folder, 'scatter_ecommerce.png'), dpi=300, bbox_inches='tight')
scatter_entertainment.savefig(os.path.join(downloads_folder, 'scatter_entertainment.png'), dpi=300, bbox_inches='tight')
heatmap.savefig(os.path.join(downloads_folder, 'heatmap.png'), dpi=300, bbox_inches='tight')
violin.savefig(os.path.join(downloads_folder, 'violin.png'), dpi=300, bbox_inches='tight')
cold_start.savefig(os.path.join(downloads_folder, 'cold_start.png'), dpi=300, bbox_inches='tight')
spider_ecommerce.savefig(os.path.join(downloads_folder, 'spider_ecommerce.png'), dpi=300, bbox_inches='tight')
spider_entertainment.savefig(os.path.join(downloads_folder, 'spider_entertainment.png'), dpi=300, bbox_inches='tight')

# For interactive plotly visualizations
bubble_ecommerce.write_html(os.path.join(downloads_folder, 'bubble_ecommerce.html'))
bubble_entertainment.write_html(os.path.join(downloads_folder, 'bubble_entertainment.html'))

print(f"All visualizations have been saved to {downloads_folder}")