#!/usr/bin/env python3
"""Export saved SMPT chain figures to editable PGFPlots; never run simulations.

No estimates/intervals/tests are recomputed. Numeric tables retain float64 values
at 17 significant digits. The sole coordinate transform is rho=.003*sigma_full,
as used by the existing descriptor renderer and manuscript.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'output/analysis/chain_corrected_figures_20261008'
DEST = ROOT / 'paper/SMPT paper/figures_pgfplots'
SIZES = (50, 100, 250, 1000)


def write_table(name, frame):
    path = DEST / name
    assert np.isfinite(frame.to_numpy(dtype=float)).all(), name
    frame.to_csv(path, sep=' ', index=False, float_format='%.17g')
    reread = pd.read_csv(path, sep=r'\s+')
    assert frame.shape == reread.shape
    np.testing.assert_allclose(frame.to_numpy(dtype=float), reread.to_numpy(dtype=float), rtol=2e-15, atol=1e-15)


def tex(name, body):
    (DEST / name).write_text(body + '\n', encoding='utf-8')


def colormap(name, cmap):
    colors = matplotlib.colormaps[cmap](np.linspace(0, 1, 11))[:, :3]
    return r'\pgfplotsset{colormap={' + name + '}{' + '\n'.join(
        'rgb=(%.8f,%.8f,%.8f);' % tuple(c) for c in colors) + '}}\n'


def heatmaps(frame, prefix, xcolumn, xvalues, shift, w1, xlabel):
    """Four panels with shared metric scales and native editable matrix plots."""
    signed_limit = max(1., float(frame[shift].abs().max()))
    w1_limit = max(1., float(frame[w1].max()))
    pieces = [r'\begin{tikzpicture}',
              r'\begin{groupplot}[paper axis,group style={group size=2 by 2,horizontal sep=3.4cm,vertical sep=1.7cm},width=.31\linewidth,height=4.25cm,scale only axis]']
    for ki, k in enumerate((1, 5)):
        rows = frame[frame.k == k].copy().sort_values(['aspl', xcolumn])
        assert len(rows) == len(xvalues)*6
        for mi, field in enumerate((shift, w1)):
            signed = mi == 0
            limit = signed_limit if signed else w1_limit
            name = f'{prefix}_k{k}_{"median" if signed else "w1"}.dat'
            table = pd.DataFrame({'x': [xvalues.index(int(v)) for v in rows[xcolumn]],
                                  'y': rows.aspl.to_numpy()-1,
                                  'value': rows[field].to_numpy()})
            table['hatch'] = ((rows.rel_w1_perm_p.to_numpy() >= .05).astype(int)
                              if prefix == 'accuracy' and not signed else 0)
            write_table(name, table)
            title = f'({"abcd"[2*ki+mi]}) $k={k}$: ' + ('median shift' if signed else 'Wasserstein distance')
            metric = (r'$b_9$ (\%)' if prefix == 'accuracy' else r'$b_c$ (\%)') if signed else r'$e_W$ (\%)'
            cellfont = r'\ArrivalCellFont' if prefix == 'chain_heatmap' else r'\scriptsize'
            # Fixed decimal strings in the dense map use the article text
            # font; omit the math wrapper (which widens the minus sign).
            number_style = 'assume math mode=true,' if prefix == 'chain_heatmap' else ''
            pieces.append(r'\nextgroupplot[' + ','.join([
                f'title={{{title}}}', f'xmin=-.5,xmax={len(xvalues)-.5:g},ymin=-.5,ymax=5.5',
                'xtick={'+','.join(str(i) for i in range(len(xvalues)))+'}',
                'xticklabels={'+','.join(str(v) for v in xvalues)+'}',
                'ytick={0,1,2,3,4,5},yticklabels={1,2,3,4,5,6}',
                f'xlabel={{{xlabel}}},ylabel={{Target internal ASPL}}',
                f'colormap name={"paperdiv" if signed else "paperseq"}',
                f'point meta min={-limit if signed else 0:.17g},point meta max={limit:.17g}',
                'colorbar' if ki == 0 else 'colorbar=false',
                'colorbar style={width=3mm,ylabel={'+metric+'},ylabel style={font=\\scriptsize},scaled y ticks=false}',
                'grid=none,tick align=outside',
            ]) + ']')
            pieces.append(r'\addplot[matrix plot*,mesh/cols=' + str(len(xvalues)) + r',point meta=explicit] table[x=x,y=y,meta=value]{figures_pgfplots/' + name + '};')
            # Labels/hatching read the same editable table as the colors.
            pieces += [r'\pgfplotstableread{figures_pgfplots/'+name+r'}\heatmapdata',
                       r'\pgfplotstableforeachcolumnelement{value}\of\heatmapdata\as\cellvalue{',
                       r'\pgfplotstablegetelem{\pgfplotstablerow}{x}\of\heatmapdata\edef\cellx{\pgfplotsretval}',
                       r'\pgfplotstablegetelem{\pgfplotstablerow}{y}\of\heatmapdata\edef\celly{\pgfplotsretval}',
                       r'\pgfplotstablegetelem{\pgfplotstablerow}{hatch}\of\heatmapdata',
                       r'\ifnum\pgfplotsretval=1\relax\edef\drawcellhatch{\noexpand\path[pattern=north east lines,pattern color=gray!45] (axis cs:{\cellx-.5},{\celly-.5}) rectangle (axis cs:{\cellx+.5},{\celly+.5});}\drawcellhatch\fi',
                       r'\pgfmathparse{abs(\cellvalue)>'+f'{.58*limit:.17g}'+'}',
                       r'\ifdim\pgfmathresult pt>0pt\def\celltextcolor{white}\else\def\celltextcolor{black}\fi',
                       # PGFPlots defers axis-coordinate processing: freeze
                       # each iteration now, or all nodes use the final cell.
                       r'\edef\drawcelllabel{\noexpand\node[font=\noexpand'+cellfont+r',text=\celltextcolor] at (axis cs:\cellx,\celly) {\noexpand\pgfmathprintnumber['+number_style+'fixed,precision=1,fixed zerofill,'+('showpos=true' if signed and prefix=='accuracy' else 'showpos=false')+r']{\cellvalue}};}\drawcelllabel',
                       '}']
            for x in range(len(xvalues)-1):
                pieces.append(r'\draw[white,line width=.4pt] (axis cs:' + f'{x+.5},-.5) -- (axis cs:{x+.5},5.5);')
            for y in range(5):
                pieces.append(r'\draw[white,line width=.4pt] (axis cs:-.5,' + f'{y+.5}) -- (axis cs:{len(xvalues)-.5},{y+.5});')
    pieces.extend([r'\end{groupplot}', r'\end{tikzpicture}'])
    tex(prefix+'.tex', '\n'.join(pieces))
    return {'median_limit': signed_limit, 'w1_limit': w1_limit, 'cells': len(frame)}


def clique_chain():
    path = SOURCE / 'trajectory_cdf_cache/aspl1_k1.npz'
    with np.load(path) as cache:
        assert int(cache['checkpoint_count']) == 400
        visible = cache['grid'] <= 6000
        for key in ('full', 'mm'):
            q = cache[key+'_quantiles']
            assert q.shape == (3, 501)
            write_table('chain_'+key+'_trajectory.dat', pd.DataFrame({
                'time': cache['grid'][visible], 'lower': q[0][visible],
                'median': q[1][visible], 'upper': q[2][visible]}))
            values = np.sort(cache[key+'_arrivals'])
            assert len(values) == 400 and np.isfinite(values).all()
            write_table('chain_'+key+'_cdf.dat', pd.DataFrame({
                'time': np.r_[0, values], 'cdf': np.r_[0, np.arange(1,401)/400]}))
    pieces = [r'\begin{tikzpicture}', r'\begin{groupplot}[paper axis,group style={group size=2 by 1,horizontal sep=1.8cm},width=.38\linewidth,height=4.6cm,scale only axis,xmin=0,xmax=6000,xtick={0,2000,4000,6000},xlabel={Time}]',
              r'\nextgroupplot[title={(a) Epidemic propagation},ylabel={Infected nodes},ymin=0,ymax=10300,ytick={0,2500,5000,7500,10000},scaled y ticks=false,legend pos=north west]']
    for key, color, label in [('full','FullBlue',r'\Full{}'),('mm','ScaleOrange',r'\TS{}')]:
        p = 'figures_pgfplots/chain_'+key+'_trajectory.dat'
        pieces += [r'\addplot[draw=none,name path='+key+'lo,forget plot] table[x=time,y=lower]{'+p+'};',
                   r'\addplot[draw=none,name path='+key+'hi,forget plot] table[x=time,y=upper]{'+p+'};',
                   r'\addplot[fill='+color+r',fill opacity=.16,draw=none,forget plot] fill between[of='+key+'lo and '+key+'hi];',
                   r'\addplot['+color+',thick] table[x=time,y=median]{'+p+'};', r'\addlegendentry{'+label+'}']
    pieces.append(r'\nextgroupplot[title={(b) Arrival in the last community},ylabel={Empirical CDF},ymin=0,ymax=1.02,ytick={0,.25,.5,.75,1}]')
    for key, color in [('full','FullBlue'),('mm','ScaleOrange')]:
        pieces.append(r'\addplot['+color+',thick,const plot] table[x=time,y=cdf]{figures_pgfplots/chain_'+key+'_cdf.dat};')
    pieces += [r'\end{groupplot}', r'\end{tikzpicture}']
    tex('chain_cliques.tex', '\n'.join(pieces))


def descriptor(summary):
    pieces = [r'\begin{tikzpicture}', r'\begin{groupplot}[paper axis,group style={group size=3 by 1,horizontal sep=1.7cm},width=.24\linewidth,height=4.1cm,scale only axis]']
    for column, yfield in enumerate(('rel_w1_pct', 'median_shift_pct')):
        pieces.append(r'\nextgroupplot[xmode=log,xlabel={$\rho=\beta\sigma_T$},ylabel={' + (r'$e_W$ of $T_9$ (\%)' if column == 0 else r'Median shift $b_9$ (\%)') + '},title={' + ('(a) Distributional discrepancy' if column==0 else '(b) Signed median shift') + '}]')
        if column == 1:
            pieces.append(r'\addplot[gray,no marks] coordinates {' + f'({summary.plotrho.min():.17g},0) ({summary.plotrho.max():.17g},0)' + '};')
        for n in SIZES:
            for k in (1, 5):
                rows = summary[(summary.n==n)&(summary.k==k)].sort_values('plotrho')
                name = f'descriptor_n{n}_k{k}.dat'
                if column == 0:
                    write_table(name, rows[['plotrho','rel_w1_pct','median_shift_pct','rel_w1_null95_pct']])
                marker = '*' if k==1 else 'triangle'
                style = f'Size{n},mark={marker},mark size=1.7pt,mark options={{fill={"Size"+str(n) if k==1 else "white"}}}'
                pieces.append(r'\addplot[' + style + '] table[x=plotrho,y='+yfield+']{figures_pgfplots/'+name+'};')
                if column==0:
                    pieces.append(r'\addplot[gray!65,only marks,mark=-,mark size=2pt] table[x=plotrho,y=rel_w1_null95_pct]{figures_pgfplots/'+name+'};')
    pieces.append(r'\nextgroupplot[title={(c) First vs later crossings},xmin=-47,xmax=5,ymin=-47,ymax=5,xtick={-40,-20,0},ytick={-40,-20,0},xlabel={First crossing: mean shift (\%)},ylabel={Crossings 2--9: mean shift (\%)}]')
    pieces.append(r'\addplot[gray,no marks] coordinates {(-47,-47) (5,5)};')
    for n in SIZES:
        for k in (1,5):
            rows = summary[(summary.n==n)&(summary.k==k)&(summary.aspl>=3)].copy()
            rows['errlo'] = rows.first_cross_shift_pct-rows.first_cross_ci_low
            rows['errhi'] = rows.first_cross_ci_high-rows.first_cross_shift_pct
            assert rows[['errlo','errhi']].min().min() >= 0
            name = f'descriptor_cross_n{n}_k{k}.dat'
            write_table(name, rows[['first_cross_shift_pct','later_cross_shift_pct','errlo','errhi']])
            pieces.append(r'\addplot[only marks,Size'+str(n)+',mark='+('*' if k==1 else 'triangle')+',mark size=1.7pt,mark options={fill='+('Size'+str(n) if k==1 else 'white')+r'},error bars/.cd,x dir=both,x explicit,error bar style={line width=.3pt},error mark options={mark size=0pt}] table[x=first_cross_shift_pct,y=later_cross_shift_pct,x error minus=errlo,x error plus=errhi]{figures_pgfplots/'+name+'};')
    pieces.append(r'\node[anchor=north west,font=\tiny,align=left,text=gray] at (axis cs:-45,3) {ASPL $\geq3$\\Below diagonal: later\\crossings more biased};')
    pieces += [r'\end{groupplot}', r'\node[anchor=north,font=\scriptsize] at ([yshift=-1.25cm]group c2r1.south) {' + r'\textcolor{Size50}{$\blacksquare$} $n=50$\quad\textcolor{Size100}{$\blacksquare$} $n=100$\quad\textcolor{Size250}{$\blacksquare$} $n=250$\quad\textcolor{Size1000}{$\blacksquare$} $n=1000$\quad$\bullet$ $k=1$\quad$\triangle$ $k=5$\quad\textcolor{gray}{---} null 95\%'+'};', r'\end{tikzpicture}']
    tex('descriptor.tex', '\n'.join(pieces))


def main():
    DEST.mkdir(exist_ok=True)
    summary = pd.read_csv(SOURCE/'cell_summary.csv')
    expected = {(n,a,k) for n in SIZES for a in range(1,7) for k in (1,5)}
    assert len(summary)==48 and set(zip(summary.n,summary.aspl,summary.k))==expected
    assert (summary.R==400).all()
    layer = pd.read_csv(SOURCE/'median_layer_metrics.csv').rename(columns={'target_ASPL':'aspl'})
    assert len(layer)==108 and (layer.paired_runs==400).all()
    assert set(zip(layer.aspl,layer.k,layer.community)) == {(a,k,c) for a in range(1,7) for k in (1,5) for c in range(1,10)}
    common = r'''% Shared plotting style; all text inherits the article font.
\providecommand{\ArrivalCellFont}{\fontsize{6}{7}\selectfont}
\pgfplotsset{compat=1.18,paper axis/.style={font=\scriptsize,label style={font=\scriptsize},title style={font=\small},tick label style={font=\scriptsize},legend style={font=\scriptsize,draw=none,fill=none},axis line style={black!65},tick style={black!65},grid=major,grid style={black!10},scaled ticks=false,unbounded coords=jump,clip=true}}
\definecolor{FullBlue}{HTML}{2878B5}
\definecolor{ScaleOrange}{HTML}{D45A44}
\definecolor{Size50}{HTML}{D73027}
\definecolor{Size100}{HTML}{FC8D59}
\definecolor{Size250}{HTML}{4575B4}
\definecolor{Size1000}{HTML}{313695}
'''
    tex('style.tex',common+colormap('paperdiv','RdBu_r')+colormap('paperseq','YlOrBr'))
    accuracy = heatmaps(summary, 'accuracy', 'n', list(SIZES), 'median_shift_pct','rel_w1_pct',r'Community size $n$')
    chain = heatmaps(layer, 'chain_heatmap', 'community', list(range(1,10)), 'signed_relative_median_shift_percent','relative_w1_percent','Community reached $c$')
    clique_chain()
    summary['plotrho'] = .003*summary.sigma_full
    descriptor(summary)
    inputs = [SOURCE/'cell_summary.csv', SOURCE/'median_layer_metrics.csv', SOURCE/'trajectory_cdf_cache/aspl1_k1.npz']
    manifest = {'purpose':'Editable LaTeX figures from immutable saved corrected-chain data',
                'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                'accuracy':accuracy,'chain_heatmap':chain,'paired_runs_per_cell':400,
                'rho':'.003*sigma_full; manuscript definition, not the alternative CSV rho column',
                'verification':'17-digit tables round-trip to source within 2e-15 relative tolerance; exact configuration coverage; no statistical recomputation or simulations',
                'uncertainty':'Clique-chain bands are IQR; descriptor horizontal intervals are saved 95% bootstrap intervals; hatches use saved W1 permutation p>=.05 only'}
    (DEST/'chain_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Exported four chain figures; 48 accuracy cells, 108 arrival cells, 400 paired terminal arrivals/model.')


if __name__ == '__main__':
    main()
