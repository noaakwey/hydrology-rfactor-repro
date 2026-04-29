import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relate annual R-factor variability to teleconnection indices.")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "output", "tables"))
    parser.add_argument("--plot-dir", default=os.path.join(ROOT, "output", "plots"))
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2024)
    return parser.parse_args()


def fetch_teleconnections(output_dir: str, plot_dir: str, start_year: int, end_year: int):
    print("Получение данных индексов NAO и SCAND с серверов NOAA CPC...")
    
    nao_url = "https://psl.noaa.gov/data/correlation/nao.data"
    scand_url = "https://psl.noaa.gov/data/correlation/scand.data"
    
    def fetch_clean_df(url):
        data = []
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode('utf-8').splitlines()
            for line in lines:
                parts = line.split()
                # PSL data usually has Year and 12 months. Sometimes header/footer exists.
                if len(parts) >= 13 and parts[0].isdigit() and int(parts[0]) > 1900:
                    year = int(parts[0])
                    vals = [float(p) if float(p) != -99.90 else np.nan for p in parts[1:13]]
                    data.append([year] + vals)
        return pd.DataFrame(data)

    nao_df = fetch_clean_df(nao_url)
    scand_df = fetch_clean_df(scand_url)
    
    # Format columns: Year, 1, 2, ..., 12
    columns = ['Year'] + list(range(1, 13))
    nao_df.columns = columns
    scand_df.columns = columns
    
    # We are interested in the R-factor from 2001 to 2024
    r_factor_path = os.path.join(output_dir, "domain_annual_rfactor.csv")
    if not os.path.exists(r_factor_path):
        print(f"File not found: {r_factor_path}")
        return
        
    r_df = pd.read_csv(r_factor_path)
    
    # Let's aggregate indices.
    # R-factor is usually driven by summer/warm-period precipitation (May-September) in this region.
    # Alternatively, we can calculate the annual mean of the indices.
    
    # NAO Summer (May-Sep)
    nao_summer = nao_df.set_index('Year').loc[start_year:end_year, 5:9].mean(axis=1)
    # SCAND Summer (May-Sep)
    scand_summer = scand_df.set_index('Year').loc[start_year:end_year, 5:9].mean(axis=1)
    
    # Annual 
    nao_annual = nao_df.set_index('Year').loc[start_year:end_year, 1:12].mean(axis=1)
    scand_annual = scand_df.set_index('Year').loc[start_year:end_year, 1:12].mean(axis=1)
    
    analysis_df = r_df.set_index('year').copy()
    analysis_df['NAO_Summer'] = nao_summer
    analysis_df['SCAND_Summer'] = scand_summer
    analysis_df['NAO_Annual'] = nao_annual
    analysis_df['SCAND_Annual'] = scand_annual
    
    # Drop rows with NaN (if 2024 is incomplete for summer indices)
    analysis_df = analysis_df.dropna()
    print("Данные для анализа собраны:")
    print(analysis_df.head())
    
    # Calculate correlations
    print("\n--- Корреляционный анализ ---")
    vars_to_test = ['NAO_Summer', 'SCAND_Summer', 'NAO_Annual', 'SCAND_Annual']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, var in enumerate(vars_to_test):
        # Pearson
        r_p, p_p = stats.pearsonr(analysis_df[var], analysis_df['r_factor'])
        # Spearman
        r_s, p_s = stats.spearmanr(analysis_df[var], analysis_df['r_factor'])
        
        print(f"\n{var}:")
        print(f"  Pearson:  r = {r_p:.3f}, p = {p_p:.3f}")
        print(f"  Spearman: r = {r_s:.3f}, p = {p_s:.3f}")
        
        # Plot
        ax = axes[i]
        ax.scatter(analysis_df[var], analysis_df['r_factor'], alpha=0.7)
        
        # Regression line
        m, b = np.polyfit(analysis_df[var], analysis_df['r_factor'], 1)
        ax.plot(analysis_df[var], m * analysis_df[var] + b, color='red', alpha=0.5)
        
        ax.set_title(f'Связь R-фактора и {var}')
        ax.set_xlabel(f'{var} индекс')
        ax.set_ylabel('R-фактор')
        ax.grid(alpha=0.3)
        
        # Add text box with correlation (in Russian)
        textstr = '\n'.join((
            f'Пирсон: $r$={r_p:.2f} ($p$={p_p:.3f})',
            f'Спирмен: $r_s$={r_s:.2f} ($p$={p_s:.3f})'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
                
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "teleconnections_correlation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nГрафик сохранен: {plot_path}")

if __name__ == "__main__":
    args = parse_args()
    fetch_teleconnections(args.output_dir, args.plot_dir, args.start_year, args.end_year)
