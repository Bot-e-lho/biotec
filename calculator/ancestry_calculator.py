import argparse
import logging
import os
import time
import gc
import pandas as pd
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

POPULATIONS_26 = {
    "YRI": "Yoruba in Ibadan, Nigéria", "LWK": "Luhya in Webuye, Quênia", "GWD": "Gambian in Western Divisions, Gâmbia",
    "MSL": "Mende in Sierra Leone", "ESN": "Esan in Nigéria", "ASW": "African Ancestry in Southwest USA",
    "ACB": "African Caribbeans in Barbados", "MXL": "Mexican Ancestry in Los Angeles, EUA",
    "PUR": "Puerto Rican in Puerto Rico", "CLM": "Colombian in Medellín, Colômbia",
    "PEL": "Peruvian in Lima, Peru", "CHB": "Han Chinese in Beijing, China",
    "JPT": "Japanese in Tokyo, Japão", "CHS": "Southern Han Chinese",
    "CDX": "Chinese Dai in Xishuangbanna, China", "KHV": "Kinh in Ho Chi Minh City, Vietnã",
    "CEU": "Utah residents with Northern/Western European ancestry (EUA)",
    "TSI": "Toscani in Itália", "FIN": "Finnish in Finlândia", "GBR": "British in Inglaterra e Escócia",
    "IBS": "Iberian populations in Espanha", "GIH": "Gujarati Indian from Houston, EUA",
    "PJL": "Punjabi from Lahore, Paquistão", "BEB": "Bengali from Bangladesh",
    "STU": "Sri Lankan Tamil from the UK", "ITU": "Indian Telugu from the UK",
}

SUPERPOP_MAP = {
    'YRI':'AFR','LWK':'AFR','GWD':'AFR','MSL':'AFR','ESN':'AFR','ASW':'AFR','ACB':'AFR',
    'MXL':'AMR','PUR':'AMR','CLM':'AMR','PEL':'AMR',
    'CHB':'EAS','JPT':'EAS','CHS':'EAS','CDX':'EAS','KHV':'EAS',
    'CEU':'EUR','TSI':'EUR','FIN':'EUR','GBR':'EUR','IBS':'EUR',
    'GIH':'SAS','PJL':'SAS','BEB':'SAS','STU':'SAS','ITU':'SAS',
}

SUPERPOP_LABELS = {
    'EUR': 'Europa (European populations)',
    'AFR': 'África (African populations)',
    'SAS': 'Sul da Ásia (South Asian populations)',
    'EAS': 'Leste Asiático (East Asian populations)',
    'AMR': 'Américas (Admixed American populations)'
}

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

def check_file_exists(path: str):
    if not os.path.exists(path):
        logging.error("Arquivo não encontrado: %s", path)
        raise FileNotFoundError(path)

def normalize_genotype(s: str):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().upper()
    s = s.replace(' ', '')
    s = s.replace('|','').replace('/','').replace('\\','').replace('-', '')
    s = ''.join(ch for ch in s if ch in ('A','C','G','T'))
    if s == '' or s in ('.','NA','NN'):
        return np.nan
    return s[:2]

def save_plot_fig(fig, outpath):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def generate_pdf_report(report_file, sample_name, results_df, predicted, b, residuals,
                        population_cols):
    sns.set_style("whitegrid")
    sns.set_palette("tab10")
    
    results_map = results_df.set_index('População')['Percentual (%)'].to_dict()
    pops = [p for p in population_cols]
    pop_percents = [results_map.get(p, 0.0) for p in pops]
    pop_df = pd.DataFrame({'População': pops, 'Percentual (%)': pop_percents})
    pop_df['Descrição'] = pop_df['População'].map(lambda p: POPULATIONS_26.get(p, 'N/A'))
    pop_df['Superpop'] = pop_df['População'].map(lambda p: SUPERPOP_MAP.get(p, 'OUT'))
    super_df = pop_df.groupby('Superpop', as_index=False)['Percentual (%)'].sum()
    super_df = super_df[super_df['Superpop'].isin(SUPERPOP_LABELS.keys())]
    super_df = super_df.sort_values(by='Percentual (%)', ascending=False).reset_index(drop=True)
    super_df['Label'] = super_df['Superpop'].map(lambda c: SUPERPOP_LABELS.get(c, c))

    intra = {}
    for sp in super_df['Superpop'].tolist():
        sub_df = pop_df[pop_df['Superpop'] == sp].copy()
        total_sp = sub_df['Percentual (%)'].sum()
        if total_sp > 0:
            sub_df['Intra (%)'] = (sub_df['Percentual (%)'] / total_sp) * 100.0
        else:
            sub_df['Intra (%)'] = 0.0
        sub_df = sub_df.sort_values('Intra (%)', ascending=False).reset_index(drop=True)
        intra[sp] = sub_df

    with PdfPages(report_file) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax0 = fig.add_subplot(2, 1, 1)
        ax0.set_title(f"Laudo de Ancestralidade\n\nAmostra: {sample_name}\nData: {datetime.today().strftime('%d/%m/%Y')}", fontsize=16)
        ax0.axis('off')
        
        ax_tabela_super = fig.add_subplot(2, 1, 2)
        ax_tabela_super.set_title('Resumo — Superpopulações', fontsize=12)
        ax_tabela_super.axis('off')
        
        table_data = super_df[['Label', 'Percentual (%)']].round(2)
        table = ax_tabela_super.table(cellText=table_data.values,
                                       colLabels=table_data.columns,
                                       loc='center',
                                       colWidths=[0.6, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.5)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        sns.barplot(x='Label', y='Percentual (%)', data=super_df, ax=ax)
        ax.set_title('Ancestralidade por Superpopulação (%)', fontsize=16)
        ax.set_ylabel('Percentual (%)')
        ax.set_xlabel('')
        for index, row in super_df.iterrows():
            ax.text(index, row['Percentual (%)'] + 1, f"{row['Percentual (%)']:.2f}%", ha='center')
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for sp in super_df['Superpop'].tolist():
            sub_df = intra[sp]
            if sub_df.empty or sub_df['Intra (%)'].sum() == 0: continue
            
            fig = plt.figure(figsize=(8.27, 11.69))
            ax_tabela_sub = fig.add_subplot(2, 1, 1)
            ax_tabela_sub.set_title(f"Composição de Ancestralidade - {SUPERPOP_LABELS.get(sp, sp)}", fontsize=16)
            ax_tabela_sub.axis('off')
            
            table_data_sub = sub_df[['População', 'Descrição', 'Intra (%)']].round(2)
            table_sub = ax_tabela_sub.table(cellText=table_data_sub.values,
                                            colLabels=table_data_sub.columns,
                                            loc='center',
                                            colWidths=[0.2, 0.5, 0.3])
            table_sub.auto_set_font_size(False)
            table_sub.set_fontsize(8)
            table_sub.scale(1.0, 1.5)

            ax_grafico_sub = fig.add_subplot(2, 1, 2)
            sns.barplot(x='População', y='Intra (%)', data=sub_df, ax=ax_grafico_sub)
            ax_grafico_sub.set_title('Composição por Subpopulação (%)', fontsize=12)
            ax_grafico_sub.tick_params(axis='x', rotation=45, ha='right')
            ax_grafico_sub.set_ylabel('Percentual intra-super (%)')
            ax_grafico_sub.set_xlabel('')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        try:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            heat_index = pop_df['População'].tolist()
            heat_cols = list(super_df['Superpop'])
            heat_mat = np.zeros((len(heat_index), len(heat_cols)), dtype=float)
            for i, pop in enumerate(heat_index):
                sp = SUPERPOP_MAP.get(pop, None)
                if sp in heat_cols:
                    col_idx = heat_cols.index(sp)
                    sub = intra.get(sp)
                    if sub is not None:
                        val = float(sub.loc[sub['População'] == pop, 'Intra (%)'].values[0]) if (sub['População'] == pop).any() else 0.0
                        heat_mat[i, col_idx] = val
            heat_df = pd.DataFrame(heat_mat, index=heat_index, columns=[SUPERPOP_LABELS[c] for c in heat_cols])
            sns.heatmap(heat_df, ax=ax, cmap='viridis', cbar_kws={'label': 'Intra-super (%)'})
            ax.set_title('Heatmap — Subpopulações × Superpopulações (intra-super %)', fontsize=16)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            logging.warning("Falha ao gerar heatmap no PDF: %s", e)

        try:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.scatter(b, predicted, s=6)
            mn = min(np.min(b), np.min(predicted)) if b.size > 0 and predicted.size > 0 else 0
            mx = max(np.max(b), np.max(predicted)) if b.size > 0 and predicted.size > 0 else 1
            ax1.plot([mn, mx], [mn, mx], '--', color='gray')
            ax1.set_xlabel('Observado (dosagem)')
            ax1.set_ylabel('Predito (A·x)')
            ax1.set_title('Predito vs Observado')
            
            ax2 = fig.add_subplot(2, 1, 2)
            sns.histplot(residuals, bins=60, kde=True, ax=ax2)
            ax2.set_xlabel('Resíduo (observado - pred)')
            ax2.set_ylabel('Contagem')
            ax2.set_title('Histograma dos Resíduos')
            
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        except Exception as e:
            logging.warning("Falha ao adicionar página predito/resíduos no PDF: %s", e)

    logging.info("PDF salvo em: %s", report_file)


def run(genotype_file: str, frequency_file: str, output_file: str,
        drop_palindromic: bool = True, test_sample_frac: float | None = None,
        do_plots: bool = True, show_plots: bool = False,
        do_report: bool = False, report_file: str = 'laudo_ancestralidade.pdf'):
    start_time = time.time()
    logging.info("Iniciando pipeline de ancestralidade NNLS")
    check_file_exists(genotype_file)
    check_file_exists(frequency_file)

    freq_header = pd.read_csv(frequency_file, nrows=0)
    freq_cols = list(freq_header.columns)
    if 'rsid' not in freq_cols or 'allele' not in freq_cols:
        raise ValueError("Arquivo de frequências deve conter colunas 'rsid' e 'allele'.")

    population_cols = [c for c in freq_cols if c in POPULATIONS_26.keys()]
    if len(population_cols) == 0:
        raise ValueError("Nenhuma coluna de população compatível encontrada no arquivo de frequência.")
    logging.info("Populações detectadas: %s", population_cols)

    geno_header = pd.read_csv(genotype_file, nrows=0)
    geno_cols = list(geno_header.columns)
    rsid_col = None
    geno_col = None
    for c in geno_cols:
        if c.lower() in ('name', 'rsid', 'snp', 'rsid_id'):
            rsid_col = c
        if 'genot' in c.lower() or '35.' in c or 'genotipo' in c.lower():
            geno_col = c
    if rsid_col is None or geno_col is None:
        logging.error("Colunas do genótipo não detectadas automaticamente. Colunas disponíveis: %s", geno_cols)
        raise ValueError("Não foi possível detectar colunas de rsid/genótipo no arquivo de genótipo.")
    logging.info("Usando coluna rsid='%s', genótipo='%s'", rsid_col, geno_col)

    usecols_freq = ['rsid','allele'] + population_cols
    frequency_df = pd.read_csv(frequency_file, usecols=usecols_freq, dtype={'rsid':str, 'allele':str})
    genotype_df = pd.read_csv(genotype_file, usecols=[rsid_col, geno_col], dtype={rsid_col: str, geno_col: str})
    genotype_df = genotype_df.rename(columns={rsid_col: 'rsid', geno_col: 'genotype'})

    t0 = time.time()
    logging.info("Normalizando genótipos...")
    genotype_df['genotype'] = genotype_df['genotype'].astype(str).apply(normalize_genotype)
    genotype_df = genotype_df[genotype_df['genotype'].notna()]
    genotype_df = genotype_df[genotype_df['genotype'].str.len() >= 1]
    genotype_df['genotype'] = genotype_df['genotype'].str[:2]

    if test_sample_frac is not None and 0 < test_sample_frac < 1.0:
        genotype_df = genotype_df.sample(frac=test_sample_frac, random_state=42).reset_index(drop=True)
        logging.info("Modo teste: usando %.2f%% dos SNPs (%d linhas)", test_sample_frac*100.0, len(genotype_df))

    geno_exp = genotype_df.copy()
    geno_exp['a1'] = geno_exp['genotype'].str[0]
    geno_exp['a2'] = geno_exp['genotype'].str[1].fillna(geno_exp['genotype'].str[0])
    alleles_df = geno_exp.melt(id_vars=['rsid'], value_name='allele')[['rsid','allele']]
    alleles_df['weight'] = 0.5
    obs_df = alleles_df.groupby(['rsid','allele'], as_index=False)['weight'].sum()
    logging.info("Genótipos processados: %d -> %d pares rsid+allele (tempo %.2fs)", len(genotype_df), len(obs_df), time.time()-t0)

    logging.info("Fazendo merge com tabela de frequências...")
    merged = pd.merge(obs_df, frequency_df, how='inner', on=['rsid','allele'])
    if merged.empty:
        raise RuntimeError("Merge resultou em vazio: verifique rsids/allelos entre arquivos.")

    if drop_palindromic:
        logging.info("Detectando/removendo SNPs palindrômicos (A/T ou C/G)...")
        freq_pairs = frequency_df.groupby('rsid')['allele'].apply(lambda arr: set(arr)).to_dict()
        pal_rsids = [rs for rs, alleles in freq_pairs.items() if alleles == set(['A','T']) or alleles == set(['C','G'])]
        if pal_rsids:
            before = merged.shape[0]
            merged = merged[~merged['rsid'].isin(pal_rsids)]
            after = merged.shape[0]
            logging.info("Removidos %d linhas correspondentes a %d SNPs palindrômicos.", before-after, len(pal_rsids))

    logging.info("Montando matriz A e vetor b...")
    b = merged['weight'].to_numpy(dtype=np.float64)
    A = merged[population_cols].to_numpy(dtype=np.float64)
    logging.info("Shapes: A=%s, b=%s", A.shape, b.shape)
    logging.info("Memória aproximada de A: %.1f MB", A.nbytes / 1024**2)

    del genotype_df, geno_exp, alleles_df, obs_df, frequency_df
    gc.collect()

    if A.shape[0] == 0 or b.size == 0:
        raise RuntimeError("Dados insuficientes após filtragem para executar NNLS.")

    logging.info("Executando NNLS (pode demorar dependendo do tamanho)...")
    t_nnls0 = time.time()
    x, rnorm = nnls(A, b)
    t_nnls1 = time.time()
    logging.info("NNLS finalizado em %.2fs (rnorm=%.6f)", t_nnls1 - t_nnls0, rnorm)

    predicted = A.dot(x) if x.size > 0 else np.zeros_like(b)
    residuals = b - predicted
    mse = np.mean(residuals**2) if residuals.size > 0 else np.nan
    rmse = np.sqrt(mse) if not np.isnan(mse) else np.nan
    ss_res = np.sum(residuals**2) if residuals.size > 0 else np.nan
    ss_tot = np.sum((b - np.mean(b))**2) if b.size > 0 else np.nan
    r2 = 1.0 - ss_res/ss_tot if (ss_tot != 0 and not np.isnan(ss_res)) else np.nan

    logging.info("Ajuste: RMSE = %.6f, R^2 = %s", rmse, ("{:.6f}".format(r2) if not np.isnan(r2) else "N/A"))

    sum_x = np.sum(x)
    if sum_x <= 0:
        logging.warning("Vetor solução x todo zero (soma == 0). Saída zerada.")
        ancestry_proportions = np.zeros_like(x)
    else:
        ancestry_proportions = (x / sum_x) * 100.0

    results_df = pd.DataFrame({
        'População': population_cols,
        'Descrição': [POPULATIONS_26.get(pop, 'N/A') for pop in population_cols],
        'Percentual (%)': ancestry_proportions
    })
    results_df['Percentual (%)'] = results_df['Percentual (%)'].round(4)
    results_df = results_df.sort_values(by='Percentual (%)', ascending=False).reset_index(drop=True)

    logging.info("Top 10 resultados:\n%s", results_df.head(10).to_string(index=False))

    results_df.to_csv(output_file, index=False)
    logging.info("Resultados salvos em: %s", output_file)

    out_dir = os.path.dirname(os.path.abspath(output_file)) or '.'
    base_name = os.path.splitext(os.path.basename(output_file))[0]

    if do_plots:
        logging.info("Gerando plots (salvando PNGs)...")
        try:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(results_df['População'].astype(str), results_df['Percentual (%)'].astype(float))
            ax1.set_xlabel("População")
            ax1.set_ylabel("Percentual (%)")
            ax1.set_title("Proporções de Ancestralidade por População")
            ax1.tick_params(axis='x', rotation=90)
            p1 = os.path.join(out_dir, f"{base_name}_ancestry_bar_all.png")
            save_plot_fig(fig1, p1)
            logging.info("Salvo: %s", p1)
        except Exception as e:
            logging.warning("Falha ao gerar bar plot completo: %s", e)

        try:
            topn = 10
            top_df = results_df.head(topn).copy()
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.bar(top_df['População'].astype(str), top_df['Percentual (%)'].astype(float))
            ax2.set_xlabel("População")
            ax2.set_ylabel("Percentual (%)")
            ax2.set_title(f"Top {topn} Populações por Proporção")
            ax2.tick_params(axis='x', rotation=45)
            p2 = os.path.join(out_dir, f"{base_name}_ancestry_bar_top{topn}.png")
            save_plot_fig(fig2, p2)
            logging.info("Salvo: %s", p2)
        except Exception as e:
            logging.warning("Falha ao gerar bar plot top: %s", e)

        try:
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            ax3.scatter(b, predicted, s=8)
            minv = min(np.min(b), np.min(predicted)) if b.size > 0 and predicted.size > 0 else 0
            maxv = max(np.max(b), np.max(predicted)) if b.size > 0 and predicted.size > 0 else 1
            ax3.plot([minv, maxv], [minv, maxv], linestyle='--')
            ax3.set_xlabel("Observado (dosagem)")
            ax3.set_ylabel("Predito (A·x)")
            ax3.set_title("Predito vs Observado")
            p3 = os.path.join(out_dir, f"{base_name}_pred_vs_obs.png")
            save_plot_fig(fig3, p3)
            logging.info("Salvo: %s", p3)
        except Exception as e:
            logging.warning("Falha ao gerar scatter predito vs observado: %s", e)

        try:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.hist(residuals, bins=60)
            ax4.set_xlabel("Residuo (observado - pred)")
            ax4.set_ylabel("Contagem")
            ax4.set_title("Histograma dos Resíduos")
            p4 = os.path.join(out_dir, f"{base_name}_residuals_hist.png")
            save_plot_fig(fig4, p4)
            logging.info("Salvo: %s", p4)
        except Exception as e:
            logging.warning("Falha ao gerar histograma de resíduos: %s", e)

        if show_plots:
            try:
                plt.figure(figsize=(8, 5))
                plt.bar(results_df['População'].astype(str), results_df['Percentual (%)'].astype(float))
                plt.xticks(rotation=90)
                plt.title("Proporções de Ancestralidade por População")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logging.warning("Falha ao exibir plots: %s", e)

    if do_report:
        sample_name = os.path.splitext(os.path.basename(genotype_file))[0]
        try:
            generate_pdf_report(report_file, sample_name, results_df, predicted, b, residuals, population_cols)
        except Exception as e:
            logging.exception("Falha ao gerar PDF: %s", e)

    logging.info("Tempo total: %.2fs", time.time() - start_time)

def parse_args():
    p = argparse.ArgumentParser(description="Estima ancestralidade via NNLS (vetorizado) e gera laudo PDF.")
    p.add_argument('--genotype', '-g', required=True, help="CSV de genótipo (ex: 23andMe), contendo rsid e coluna de genótipo.")
    p.add_argument('--freq', '-f', required=True, help="CSV de frequências (colunas: rsid, allele, POP1, POP2, ...).")
    p.add_argument('--output', '-o', default='resultado_ancestralidade_otimizado.csv', help="Arquivo CSV de saída.")
    p.add_argument('--no-drop-pal', dest='drop_pal', action='store_false', help="Não remover SNPs palindrômicos (A/T, C/G).")
    p.add_argument('--sample', type=float, default=None, help="Amostra fracionária para teste (0.1 = usar 10%% dos SNPs).")
    p.add_argument('--plots', action='store_true', help="Gerar plots PNG (bar, top10, pred_vs_obs, residuals).")
    p.add_argument('--show-plots', action='store_true', help="Exibir plots (útil em ambiente com GUI).")
    p.add_argument('--report', action='store_true', help="Gerar laudo PDF (várias páginas, similar ao exemplo).")
    p.add_argument('--report-file', default='laudo_ancestralidade.pdf', help="Caminho do PDF de saída.")
    p.add_argument('--verbose', '-v', action='store_true', help="Modo verboso (debug).")
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    setup_logging(args.verbose)
    try:
        run(
            genotype_file=args.genotype,
            frequency_file=args.freq,
            output_file=args.output,
            drop_palindromic=args.drop_pal,
            test_sample_frac=args.sample,
            do_plots=args.plots,
            show_plots=args.show_plots,
            do_report=args.report,
            report_file=args.report_file
        )
    except Exception as e:
        logging.exception("Erro durante execução: %s", e)
        raise
