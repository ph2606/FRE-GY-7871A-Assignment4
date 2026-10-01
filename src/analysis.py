"""Session-safe Granger tests and descriptive media-comparison estimates."""
import json
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from .config import ROOT, START, CUTOFF, SEED, TICKERS

US_MAJOR = {'The New York Times','The Washington Post','CNN','Fox News','NBC News','ABC News','CBS News',
 'NPR','The Wall Street Journal','CNBC','Bloomberg','Business Insider','Fortune','WIRED','Wired','Axios',
 'The Hill','POLITICO','Politico','USA Today','USA TODAY','Newsweek','Los Angeles Times','AP News','Associated Press'}
INTL_MAJOR = {'BBC','BBC News','The Guardian','Reuters','Financial Times','The Economist','Al Jazeera',
 'The Independent','The Telegraph','The Times','CBC','CBC News','The Globe and Mail','ABC News (Australia)',
 'The Sydney Morning Herald','The Australian','South China Morning Post'}
PUBLISHER_ALIASES = {'Bloomberg.com':'Bloomberg', 'WSJ':'The Wall Street Journal',
 'businessinsider.com':'Business Insider', 'ABC News - Breaking News, Latest News and Videos':'ABC News',
 'apnews.com':'AP News','Associated Press':'AP News','Wired':'WIRED','Politico':'POLITICO',
 'USA Today':'USA TODAY','BBC':'BBC News'}

def session_assign(times, sessions):
    """Assign a publication to the first cash close at or after its UTC time."""
    closes=pd.DatetimeIndex(sessions).tz_localize('America/New_York')+pd.Timedelta(hours=16)
    utc=pd.to_datetime(times,utc=True,format='mixed')
    idx=np.searchsorted(closes.tz_convert('UTC').asi8,utc.array.asi8,side='left')
    result=pd.Series(pd.NaT,index=times.index,dtype='datetime64[ns]')
    valid=(idx<len(closes)) & utc.notna().to_numpy()
    result.loc[valid]=pd.DatetimeIndex(sessions)[idx[valid]].values
    return result

def build_design(frame,y,x,lag=1):
    """Shift on full trading calendar, then drop incomplete regression rows."""
    table=pd.DataFrame({'y':frame[y]},index=frame.index)
    for k in range(1,lag+1):
        table[f'y{k}']=frame[y].shift(k)
        table[f'x{k}']=frame[x].shift(k)
    return table.dropna()

def granger(frame,y,x,lag=1,bootstrap=4999):
    d=build_design(frame,y,x,lag)
    row={'outcome':y,'predictor':x,'lag':lag,'n':len(d),'status':'ok'}
    if len(d)<=2*lag+4:
        return {**row,'status':'insufficient_complete_rows'}
    yr=d.y.to_numpy()
    restricted=sm.add_constant(d[[f'y{k}' for k in range(1,lag+1)]],has_constant='add')
    full=sm.add_constant(d[[f'y{k}' for k in range(1,lag+1)]+[f'x{k}' for k in range(1,lag+1)]],has_constant='add')
    if np.linalg.matrix_rank(full)<full.shape[1]:return {**row,'status':'rank_deficient'}
    fit=sm.OLS(yr,full).fit(); null=sm.OLS(yr,restricted).fit()
    f,p,dfdiff=fit.compare_f_test(null)
    row.update(f=float(f),p=float(p),df_resid=int(fit.df_resid),
               coefficient_lag1=float(fit.params.x1),
               effect_per_01=float(fit.params.x1*0.1),
               ci_low_per_01=float(fit.conf_int().loc['x1',0]*0.1),
               ci_high_per_01=float(fit.conf_int().loc['x1',1]*0.1),
               coefficient_sum=float(sum(fit.params[f'x{k}'] for k in range(1,lag+1))),
               r_squared=float(fit.rsquared),
               ljung_box_p=float(acorr_ljungbox(fit.resid,lags=[1],return_df=True).lb_pvalue.iloc[0]),
               first_date=str(d.index.min().date()),last_date=str(d.index.max().date()))
    if bootstrap:
        rng=np.random.default_rng(SEED)
        # Restricted fixed-design wild bootstrap: lagged regressors are held fixed.
        xr=np.asarray(restricted); xu=np.asarray(full)
        mr=np.eye(len(d))-xr@np.linalg.pinv(xr); mu=np.eye(len(d))-xu@np.linalg.pinv(xu)
        residual=np.asarray(null.resid)/np.sqrt(np.maximum(np.diag(mr),1e-8))
        residual=residual-residual.mean()
        signs=rng.choice([-1,1],size=(len(d),bootstrap))
        ys=np.asarray(null.fittedvalues)[:,None]+residual[:,None]*signs
        ssr=((mr@ys)**2).sum(axis=0);ssu=((mu@ys)**2).sum(axis=0)
        fb=((ssr-ssu)/lag)/(ssu/fit.df_resid)
        row['wild_p']=(1+int((fb>=f).sum()))/(bootstrap+1)
        row['bootstrap_draws']=bootstrap
    return row

def aggregate(group):
    return pd.Series({'n':len(group),'sentiment':group.vader.mean(),'negative_share':group.negative.mean(),
                      'hu_negative':group.hu_negative.mean(),'risk_share':group.risk.mean()})

def analyze():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    data=pd.read_csv(ROOT/'data/processed/scored_documents.csv',low_memory=False)
    data=data.loc[data.included_analysis].copy()
    data['original_source']=data.source
    data['source']=data.source.replace(PUBLISHER_ALIASES)
    data['date']=pd.to_datetime(data.date)
    data['medium']=np.where(data.kind.eq('news'),'News headlines',np.where(data.is_comment,'HN comments','HN stories'))
    market=pd.read_csv(ROOT/'data/processed/market_daily.csv',index_col='date',parse_dates=True)
    published=pd.to_datetime(data.published_at,utc=True,format='mixed')
    available=pd.to_datetime(data.get('conservative_available_at',pd.Series(pd.NaT,index=data.index)),utc=True,format='mixed',errors='coerce')
    data['available_at']=available.fillna(published)
    data['session']=session_assign(data.available_at,market.index)
    language_path=ROOT/'data/processed/language_audit.csv'
    if language_path.exists():
        language=pd.read_csv(language_path,usecols=['id','english_probability']).drop_duplicates('id')
        data=data.merge(language,on='id',how='left',validate='one_to_one')
    data.to_csv(out/'documents_with_sessions.csv',index=False)
    rows=[]
    for (medium,date),g in data.groupby(['medium','date']):
        for subset,f in [('all_AI',g),('risk',g.loc[g.risk]),('other_AI',g.loc[~g.risk])]:
            if len(f):rows.append({'medium':medium,'date':date,'subset':subset,**aggregate(f)})
    calendar=pd.DataFrame(rows)
    calendar.to_csv(out/'calendar_sentiment.csv',index=False)
    session=market.copy()
    means={}
    for medium,prefix in [('News headlines','news'),('HN comments','social'),('HN stories','stories')]:
        f=data.loc[data.medium.eq(medium)&data.session.notna()]
        for risk in [False,True]:
            z=f.loc[f.risk] if risk else f
            label=prefix+('_risk' if risk else '')
            a=z.groupby('session').vader.mean().reindex(market.index)
            session[label]=a
            session[label+'_n']=z.groupby('session').size().reindex(market.index,fill_value=0)
            session[label+'_hu']=z.groupby('session').hu_negative.mean().reindex(market.index)
        # One vote per thread/author within each session is a composition sensitivity.
        if medium=='HN comments':
            z=f.loc[f.risk]
            session['social_risk_thread']=z.groupby(['session','thread_id']).vader.mean().groupby('session').mean().reindex(market.index)
            session['social_risk_author']=z.groupby(['session','author']).vader.mean().groupby('session').mean().reindex(market.index)
            strict=z.included_strict_sensitivity.astype(str).str.lower().isin(['true','1'])
            session['social_risk_strict']=z.loc[strict].groupby('session').vader.mean().reindex(market.index)
        if medium=='News headlines':
            z=f.loc[f.risk]
            session['news_risk_outlet']=z.groupby(['session','source']).vader.mean().groupby('session').mean().reindex(market.index)
    session['balanced_risk']=session[['news_risk','social_risk']].mean(axis=1,skipna=False)
    session['balanced']=session[['news','social']].mean(axis=1,skipna=False)
    session['balanced_risk_hu']=session[['news_risk_hu','social_risk_hu']].mean(axis=1,skipna=False)
    if 'english_probability' in data:
        e=data.loc[data.risk & data.english_probability.ge(.9)]
        ens=e.loc[e.medium.eq('News headlines')].groupby('session').vader.mean().reindex(market.index)
        ehs=e.loc[e.medium.eq('HN comments')].groupby('session').vader.mean().reindex(market.index)
        session['english_risk']=(ens+ehs)/2
        session['d_english_risk']=session.english_risk.diff()
    levels=['balanced_risk','balanced','news_risk','social_risk','news','social','stories',
            'social_risk_thread','social_risk_author','social_risk_strict','news_risk_outlet','balanced_risk_hu']
    for col in levels:session['d_'+col]=session[col].diff()
    session.to_csv(out/'session_panel.csv',index_label='date')
    # Six planned core forward tests, Holm-adjusted together.
    tests=[]
    for y in ['basket','spread','spread_qqq']:
        for lag in [1,2]:
            row=granger(session,y,'d_balanced_risk',lag)
            row['family']='core_forward';tests.append(row)
    core=pd.DataFrame(tests)
    ok=core.status.eq('ok')
    core.loc[ok,'p_holm']=multipletests(core.loc[ok,'p'],method='holm')[1]
    core.to_csv(out/'granger_core.csv',index=False)
    robust=[]
    for x in ['d_balanced','d_news_risk','d_social_risk','d_news','d_social','d_stories',
              'd_social_risk_thread','d_social_risk_author','d_social_risk_strict','d_news_risk_outlet','d_balanced_risk_hu']:
        row=granger(session,'spread',x,1);row['specification']=x;robust.append(row)
    row=granger(session,'d_balanced_risk','spread',1);row['specification']='reverse: spread -> sentiment change';robust.append(row)
    z=session.copy();z['delayed']=z.d_balanced_risk.shift(1)
    row=granger(z,'spread','delayed',1);row['specification']='one extra session delay';robust.append(row)
    z=session.copy();z.loc[z.balanced_risk.first_valid_index(),'balanced_risk']=np.nan;z['d_balanced_risk']=z.balanced_risk.diff()
    row=granger(z,'spread','d_balanced_risk',1);row['specification']='exclude first observed balanced sentiment';robust.append(row)
    row=granger(session,'buy_hold','d_balanced_risk',1);row['specification']='buy-and-hold basket';robust.append(row)
    if 'd_english_risk' in session:
        row=granger(session,'spread','d_english_risk',1);row['specification']='English detector score >= 0.90';robust.append(row)
    robust=pd.DataFrame(robust);ok=robust.status.eq('ok')
    robust.loc[ok,'p_holm']=multipletests(robust.loc[ok,'p'],method='holm')[1]
    robust.to_csv(out/'granger_sensitivity.csv',index=False)
    # Influence deletes one regression equation, preserving every lag calendar.
    design=build_design(session,'spread','d_balanced_risk',1)
    influence=[]
    for date in design.index:
        d=design.drop(index=date)
        f=sm.OLS(d.y,sm.add_constant(d[['y1','x1']])).fit()
        influence.append({'omitted_date':date,'effect_per_01':f.params.x1*.1,'p':f.pvalues.x1})
    pd.DataFrame(influence).to_csv(out/'leave_one_out.csv',index=False)
    diagnostic=[]
    for col in ['balanced_risk','d_balanced_risk','basket','spread']:
        series=session[col]
        first,last=series.first_valid_index(),series.last_valid_index()
        a=series.loc[first:last] if first is not None else series.iloc[:0]
        diagnostic_row={'variable':col,'adf_stat':np.nan,'p':np.nan,'n':int(a.notna().sum()),
                        'note':'Zero augmentation lags fixed for short sample; low power.'}
        if a.isna().any():
            diagnostic_row.update(status='skipped_internal_gaps',
                                  note='Internal missing sessions remain missing; no compressed-calendar ADF test.')
        elif len(a)<4 or a.nunique()<2:
            diagnostic_row.update(status='insufficient_or_constant_series')
        else:
            adf=adfuller(a,maxlag=0,autolag=None,regression='c')
            diagnostic_row.update(adf_stat=adf[0],p=adf[1],n=adf[3],status='ok')
        diagnostic.append(diagnostic_row)
    pd.DataFrame(diagnostic).to_csv(out/'stationarity.csv',index=False)
    bias_analysis(data,out)
    levels=pd.read_csv(ROOT/'data/processed/market_levels.csv',index_col='date',parse_dates=True)
    prices=levels.pivot_table(index=levels.index,columns='ticker',values='adjusted_close')
    baseline_date=prices.index[prices.index < market.index.min()].max()
    prices=prices.reindex(pd.DatetimeIndex([baseline_date]).union(market.index))
    stock_returns=prices.pct_change(fill_method=None).reindex(market.index)
    basket_checks=[]
    for excluded in TICKERS:
        r=stock_returns[[t for t in TICKERS if t!=excluded]].mean(axis=1,skipna=False)
        basket_checks.append({'excluded':excluded,'return_pct':100*((1+r).prod(skipna=False)-1)})
    pd.DataFrame(basket_checks).to_csv(out/'basket_exclusions.csv',index=False)
    trends=[]
    for medium,g in calendar.loc[calendar.subset.eq('risk')].groupby('medium'):
        # Equal day weighting prevents a news-volume spike dominating endpoints.
        first=g.loc[g.date.between('2026-09-08','2026-09-14')]
        last=g.loc[g.date.between('2026-09-22',CUTOFF)]
        fit=sm.OLS(g.sentiment,sm.add_constant((g.date-g.date.min()).dt.days)).fit(cov_type='HAC',cov_kwds={'maxlags':2})
        trends.append({'medium':medium,'early_mean':first.sentiment.mean(),'late_mean':last.sentiment.mean(),
                       'change':last.sentiment.mean()-first.sentiment.mean(),'early_negative_share':first.negative_share.mean(),
                       'late_negative_share':last.negative_share.mean(),'slope_per_day':fit.params.iloc[1],
                       'slope_hac_p':fit.pvalues.iloc[1],'early_days':len(first),'late_days':len(last)})
    trends=pd.DataFrame(trends)
    trends['slope_holm_p']=multipletests(trends.slope_hac_p,method='holm')[1]
    trends.to_csv(out/'trends.csv',index=False)
    summary={'cutoff':CUTOFF,'sessions':len(session),'final_day_partial_calendar':True,
             'text_window_end':'2026-09-30T16:00:00-04:00',
             'window_definition':'Requested dates through final US cash-market close; subsequent text is outside the return window.',
             'documents':data.groupby('medium').size().to_dict(),
             'risk_documents':data.loc[data.risk].groupby('medium').size().to_dict(),
             'returns':{k:100*(session[k+'_wealth'].iloc[-1]-1) for k in ['basket','spy','qqq','buy_hold']},
             'postclose_records_not_in_market_regression':int(data.session.isna().sum()),
             'core_family_size':len(core),'sensitivity_family_size':len(robust),
             'news_sources':int(data.loc[data.kind.eq('news'),'source'].nunique())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print(core.round(5).to_string(index=False))
    return session

def bias_analysis(data,out):
    summary=[]
    for (medium,risk),f in data.groupby(['medium','risk']):
        summary.append({'medium':medium,'risk':risk,**aggregate(f)})
    pd.DataFrame(summary).to_csv(out/'bias_descriptive.csv',index=False)
    news=data.loc[data.kind.eq('news')].copy()
    news['source_group']=np.select([news.source.isin(US_MAJOR),news.source.isin(INTL_MAJOR)],['US major','International major'],default='Other / unclassified')
    news['log_words']=np.log(news.n_words)
    news['date_str']=news.date.astype(str)
    news.groupby(['source','source_group']).size().rename('n').reset_index().to_csv(out/'news_source_groups.csv',index=False)
    groups=[]
    for (source_group,risk),f in news.groupby(['source_group','risk']):
        groups.append({'source_group':source_group,'risk':risk,**aggregate(f)})
    pd.DataFrame(groups).to_csv(out/'news_group_bias.csv',index=False)
    # Descriptive model uses same unit (headlines), broad AI corpus and topic/day/length controls.
    z=news.loc[news.source_group.ne('Other / unclassified')].copy()
    z['us_major']=(z.source_group=='US major').astype(int)
    z['hu_pct']=100*z.hu_negative
    models=[]
    if z.source.nunique()>=10 and z.us_major.nunique()==2:
        for target in ['negative','hu_pct']:
            fit=smf.ols(f'{target} ~ us_major + C(topic) + log_words + I(log_words**2) + C(date_str)',data=z).fit(cov_type='cluster',cov_kwds={'groups':z.source})
            interval=fit.conf_int().loc['us_major']
            models.append({'model':target,'term':'US major vs international major','n':int(fit.nobs),'clusters':z.source.nunique(),
                           'coefficient':fit.params.us_major,'se':fit.bse.us_major,'p':fit.pvalues.us_major,
                           'ci_low':interval.iloc[0],'ci_high':interval.iloc[1], 'covariance':'publisher-clustered'})
    models=pd.DataFrame(models)
    if len(models):models['p_holm']=multipletests(models.p,method='holm')[1]
    models.to_csv(out/'bias_regressions.csv',index=False)
    # Risk lexicon itself contains negative vocabulary; this is descriptive, never causal.
    topic=data.groupby(['medium','topic']).agg(n=('id','size'),vader=('vader','mean'),negative=('negative','mean'),hu_negative=('hu_negative','mean')).reset_index()
    topic.to_csv(out/'topic_summary.csv',index=False)
    # Retrieval-time points on stories only: popularity, not pre-publication information.
    stories=data.loc[data.medium.eq('HN stories')].copy()
    stories['score_numeric']=pd.to_numeric(stories.score,errors='coerce')
    stories=stories.loc[stories.score_numeric.notna()].copy()
    stories['high_engagement']=stories.score_numeric.ge(5)
    eng=stories.groupby('high_engagement').agg(n=('id','size'),risk_share=('risk','mean'),negative_share=('negative','mean'),mean_sentiment=('vader','mean')).reset_index()
    eng.to_csv(out/'engagement_bias.csv',index=False)

if __name__=='__main__':analyze()
