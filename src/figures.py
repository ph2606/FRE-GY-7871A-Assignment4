"""Compact standalone report figures and notebook PNGs."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .config import ROOT, CUTOFF

NAVY='#17334d'; TEAL='#168c8c'; GOLD='#bb8540'; GREY='#6e7987'
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,
 'axes.spines.right':False,'axes.labelcolor':NAVY,'text.color':NAVY,'axes.titleweight':'bold',
 'grid.alpha':.20,'savefig.bbox':'tight','figure.facecolor':'white'})

def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(ROOT/f'outputs/{name}.{ext}',dpi=170)
    plt.close(fig)

def build():
    out=ROOT/'outputs'
    panel=pd.read_csv(out/'session_panel.csv',index_col='date',parse_dates=True)
    daily=pd.read_csv(out/'calendar_sentiment.csv',parse_dates=['date'])
    fig,ax=plt.subplots(figsize=(6.8,2.55))
    for col,label,color in [('basket','AI basket',NAVY),('spy','SPY',GREY),('qqq','QQQ',TEAL)]:
        series=100*(panel[col+'_wealth']-1)
        ax.plot(series.index,series,label=f'{label} ({series.iloc[-1]:+.2f}%)',color=color,lw=1.8)
    ax.axhline(0,color=GREY,lw=.7);ax.set_ylabel('Cumulative return (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('Sep %d'));ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.grid(axis='y');ax.legend(frameon=False,ncol=3,loc='upper left',fontsize=8)
    ax.set_title('Adjusted-close returns from the September 4 baseline',loc='left',fontsize=10)
    save(fig,'market_returns')
    fig,axes=plt.subplots(3,1,figsize=(6.8,4.35),sharex=True,gridspec_kw={'height_ratios':[1,1,.65]})
    for medium,color,ax in [('News headlines',NAVY,axes[0]),('HN comments',TEAL,axes[1])]:
        z=daily.loc[daily.medium.eq(medium)&daily.subset.eq('risk')].sort_values('date').set_index('date')
        ax.plot(z.index,z.sentiment,color=color,alpha=.35,lw=.9,marker='o',ms=2.3,label='Daily')
        ax.plot(z.index,z.sentiment.rolling(3,min_periods=1).mean(),color=color,lw=2,label='Trailing 3-day mean')
        ax.axhline(0,color=GREY,lw=.6);ax.set_ylim(-.45,.30);ax.grid(axis='y');ax.set_ylabel('VADER compound')
        ax.set_title(medium+' · AI-risk subset',loc='left',fontsize=9)
        if ax==axes[0]:ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=7)
        axes[2].plot(z.index,z.n,label=medium,color=color,lw=1.1)
    axes[2].set_yscale('log');axes[2].set_ylabel('Risk texts\n(log scale)');axes[2].legend(frameon=False,ncol=2,fontsize=7,loc='upper right')
    for ax in axes:
        ax.axvspan(pd.Timestamp(CUTOFF),pd.Timestamp(CUTOFF)+pd.Timedelta(days=1),color=GREY,alpha=.08)
        ax.set_xlim(pd.Timestamp('2026-09-08'),pd.Timestamp('2026-09-30'))
    axes[2].xaxis.set_major_locator(mdates.DayLocator(interval=4));axes[2].xaxis.set_major_formatter(mdates.DateFormatter('Sep %d'))
    axes[2].set_xlabel('Calendar publication date · Sep 30 text ends at 16:00 New York')
    fig.tight_layout(h_pad=.8);save(fig,'sentiment_trends')
    desc=pd.read_csv(out/'bias_descriptive.csv')
    fig,ax=plt.subplots(figsize=(6.8,2.25));x=np.arange(3)
    for risk,offset,label,color in [(False,-.17,'Other AI',TEAL),(True,.17,'Risk / safety AI',NAVY)]:
        z=desc.loc[desc.risk.eq(risk)].set_index('medium').reindex(['News headlines','HN comments','HN stories'])
        bars=ax.bar(x+offset,100*z.negative_share,width=.32,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f%%',padding=2,fontsize=8)
    ax.set_xticks(x,['News headlines','HN comments','HN stories']);ax.set_ylim(0,78);ax.set_ylabel('VADER-negative texts (%)')
    ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=8);ax.grid(axis='y');ax.set_axisbelow(True)
    save(fig,'bias_comparison')
    inf=pd.read_csv(out/'leave_one_out.csv',parse_dates=['omitted_date'])
    core=pd.read_csv(out/'granger_core.csv');main=core.loc[core.outcome.eq('spread')&core.lag.eq(1)].iloc[0]
    fig,ax=plt.subplots(figsize=(6.8,1.9))
    ax.scatter(inf.omitted_date,inf.effect_per_01,color=TEAL,s=24)
    ax.axhline(main.effect_per_01,color=NAVY,lw=1,label=f'All {int(main.n)} equations')
    ax.axhline(0,color=GREY,lw=.7);ax.set_ylabel('Effect per +0.1\nsentiment (pp)');ax.set_xlabel('Omitted return equation')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'));ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
    ax.legend(frameon=False,fontsize=8);ax.grid(axis='y');save(fig,'influence')

if __name__=='__main__':build()
