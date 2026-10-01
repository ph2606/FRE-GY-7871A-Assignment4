"""Shared report/notebook narrative populated only from computed exhibits."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
REPO='https://github.com/ph2606/FRE-GY-7871A-Assignment4'

def number(x,d=2):return f'{float(x):,.{d}f}'
def pct(x,d=1):return f'{100*float(x):.{d}f}%'
def table(headers,rows,widths=None):return {'type':'table','headers':headers,'rows':rows,'widths':widths}
def paragraph(text):return {'type':'p','text':text}
def figure(name,caption,width='.98'):return {'type':'figure','name':name,'caption':caption,'width':width}

def sections():
    s=json.loads((OUT/'summary.json').read_text())
    end=pd.Timestamp(s['cutoff'])
    end_label=f'September {end.day}'
    window=f'September 8–{end.day}, 2026'
    core=pd.read_csv(OUT/'granger_core.csv');rob=pd.read_csv(OUT/'granger_sensitivity.csv')
    main=core.loc[core.outcome.eq('spread')&core.lag.eq(1)].iloc[0]
    second_lag=core.loc[core.outcome.eq('spread')&core.lag.eq(2)].iloc[0]
    desc=pd.read_csv(OUT/'bias_descriptive.csv').set_index(['medium','risk'])
    trends=pd.read_csv(OUT/'trends.csv').set_index('medium')
    bias=pd.read_csv(OUT/'bias_regressions.csv').set_index('model')
    eng=pd.read_csv(OUT/'engagement_bias.csv').set_index('high_engagement')
    constituents=pd.read_csv(ROOT/'data/processed/constituents.csv').set_index('ticker')
    exclusion=pd.read_csv(OUT/'basket_exclusions.csv').set_index('excluded')
    loo=pd.read_csv(OUT/'leave_one_out.csv').set_index('omitted_date')
    social=json.loads((OUT/'social_audit.json').read_text())
    news=json.loads((OUT/'news_manifest.json').read_text())
    filt=pd.read_csv(ROOT/'data/processed/analysis_filter_audit.csv').fillna('')
    freeze=pd.Timestamp(news['cutoff_utc']).tz_convert('America/New_York')
    social_freeze=pd.Timestamp(social['cutoff_utc']).tz_convert('America/New_York')
    final_close=pd.Timestamp('2026-09-30 16:00',tz='America/New_York')
    if end.date()!=final_close.date() or freeze!=final_close or social_freeze!=final_close:
        raise ValueError('Final report requires analysis and both text corpora through the September 30 16:00 New York close.')
    freeze_label=f'{freeze:%H:%M} New York time on September {freeze.day}'
    nex=news['exclusions']
    sex={r['stage']:r['removed_at_stage'] for r in social['filter_flow']}
    nd=nex.get('duplicate normalized headline within publisher',0)
    na=nex.get('no explicit AI term in headline',0)
    sd=sex.get('duplicate_normalized_text',0)
    sa=sex.get('no_ai_match_in_own_text',0)
    core_pass=int(core.p_holm.lt(.05).sum())
    sensitivity_pass=int(rob.p_holm.lt(.05).sum())
    slope_pass=int(trends.slope_holm_p.lt(.05).sum())
    core_result=('No core Granger test survives Holm correction at 5%.' if not core_pass else
                 f'{core_pass} of {len(core)} core Granger tests survive Holm correction at 5%; short-sample and robustness limits still apply.')
    slope_result=('None of the time slopes survives Holm correction at 5%.' if not slope_pass else
                  f'{slope_pass} of {len(trends)} time slopes survive Holm correction at 5%.')
    if .045 <= trends.loc['News headlines','slope_holm_p'] <= .055:
        slope_result+=f' The news slope is borderline (adjusted p={trends.loc["News headlines","slope_holm_p"]:.6f}), so the trend should be read descriptively.'
    sensitivity_result=('None survives its sensitivity-family Holm correction.' if not sensitivity_pass else
                        f'{sensitivity_pass} of {len(rob)} tests survive sensitivity-family Holm correction.')
    b=bias.loc['negative'];h=bias.loc['hu_pct']
    headline_direction='rose' if trends.loc['News headlines','change']>0 else 'fell'
    bias_direction='opposite to' if b.coefficient<0 else 'the same as'
    measure_agreement=(b.p_holm<.05 and h.p_holm<.05 and b.coefficient*h.coefficient>0)
    measure_conclusion=('The direction is supported by both measures.' if measure_agreement else
                        'The two measures do not provide equally strong evidence for this source-group difference.')
    influence_date=(loo.effect_per_01-main.effect_per_01).abs().idxmax()
    influence=loo.loc[influence_date]
    influence_label=f'September {pd.Timestamp(influence_date).day}'
    nnews=s['documents']['News headlines'];ncomm=s['documents']['HN comments'];nstory=s['documents']['HN stories']
    risknews=s['risk_documents']['News headlines'];riskcomm=s['risk_documents']['HN comments']
    beta=main.effect_per_01
    classes={'NVDA':'Accelerators','MSFT':'Cloud / models','GOOGL':'Cloud / models','AMZN':'Cloud / models','META':'Models / platforms','AVGO':'AI infrastructure'}
    basketrows=[[ticker,classes[ticker],'16.67%',number(constituents.loc[ticker,'cumulative_return_pct'])+'%'] for ticker in classes]
    first=[
      paragraph('**Research question.** Did the AI-risk discussion following Evan Hubinger’s September 8 X post predict returns in AI equities, and did coverage display a negativity pattern comparable to the assigned COVID-news study? The post motivates the window; the analysis does not identify its isolated causal effect. [1, 2]'),
      paragraph(f'**Observed answer.** Risk-headline mean sentiment {headline_direction} between the opening and closing periods. The six-stock basket returned **{number(s["returns"]["basket"])}%**, versus **{number(s["returns"]["spy"])}% for SPY** and **{number(s["returns"]["qqq"])}% for QQQ**. {core_result} The US-versus-international negativity coefficient points in the direction {bias_direction} the COVID paper’s finding. {measure_conclusion}'),
      paragraph(f'**Observation window.** {window}, covering {s["sessions"]} completed market sessions. Text is frozen at {freeze_label}, the final market close. September 30 text after 16:00 is outside this aligned study; the final calendar day is therefore partial. No future observation or missing return is filled.'),
      table(['Ticker','AI exposure used to select basket','Initial / target weight','Window return'],basketrows,['.11','.44','.22','.17']),
      paragraph(r'Weights are fixed at $1/6$ before examining returns. Each session’s simple basket return is the mean of six dividend/split-adjusted returns: $r^B_t=100\sum_i w_i(P^*_{i,t}/P^*_{i,t-1}-1)$. Rebalancing is at the preceding close; compounded returns exclude fees. September 4 supplies the baseline because September 7 was Labor Day. SPY is the broad benchmark; QQQ is the technology-heavy comparison. The sample is deliberately an AI-exposed basket, not a pure-play AI industry index. [8]'),
      figure('market_returns',f'Figure 1. Adjusted-close cumulative returns, September 8–{end.day}. Lines describe realized returns, not returns caused by the X post.','.98'),
    ]
    nf=filt.loc[filt.kind.eq('news')&filt.analysis_exclusion.ne(''),'n'].sum()
    sf=filt.loc[filt.kind.eq('social')&filt.analysis_exclusion.ne(''),'n'].sum()
    second=[
      paragraph('Collection begins with **broad AI terms**, not “danger,” “doom,” or negative stock returns. Four fixed Google News query families cover artificial intelligence, OpenAI/ChatGPT, Anthropic/Claude, and AI with basket-company names; the Guardian AI archive supplies publisher pages. Nine Hacker News (HN) query terms cover AI, artificial intelligence, LLM, ChatGPT, OpenAI, Anthropic, superintelligence, machine learning and AGI. Daily HN queries are recursively split in time to resolve the 1,000-result ceiling. [6, 7]'),
      table(['Stage','News','Hacker News'],[
       ['Retrieved hits / unique HN IDs',f'{news["raw_discovered_rows"]:,} rows',f'{social["unique_candidate_ids"]:,} IDs'],
       ['Duplicate removals in collection',f'{nd:,}',f'{sd:,}'],
       ['Own headline / text lacks AI anchor',f'{na:,}',f'{sa:,}'],
       ['Other collection exclusions',f'{sum(nex.values())-nd-na:,}',f'{sum(sex.values())-sd-sa:,}'],
       ['After collection',f'{news["included_headlines"]:,}',f'{social["clean_records"]:,}'],
       ['Additional analysis exclusions',f'{int(nf):,}',f'{int(sf):,}'],
       ['Final analysis corpus',f'{nnews:,} headlines',f'{ncomm:,} comments + {nstory:,} stories'],
       ['Risk / safety subset',f'{risknews:,} headlines',f'{riskcomm:,} comments + {s["risk_documents"]["HN stories"]:,} stories'],
      ],['.49','.20','.26']),
      paragraph(f'News collection excludes {nex.get("outside frozen information window",0):,} out-of-window items, {nex.get("live blog or audiovisual/gallery format",0):,} live/audiovisual items, {nex.get("headline shorter than four words",0):,} short headlines and {nex.get("publisher modification after frozen cutoff",0):,} items modified after cutoff. HN excludes {sex.get("not_story_or_comment",0):,} non-story/comment items, {sex.get("fewer_than_three_words",0):,} texts under three words and {sex.get("link_spam_rule",0):,} link-spam cases; the notebook checks the filter reconciliation. Analysis excludes known nonmedia sources, then requires five words and an AI anchor in the scored text. Exact normalized text duplicates retain the earliest eligible availability within each medium. No fuzzy matching or verified-bot classifier is claimed.'),
      paragraph(f'**Reach and automation.** Followers, views and comment scores are unavailable in HN; no fictional thresholds are applied. Story points and comment counts are retrieval-time snapshots. A sensitivity sample excludes story points below 5, bot-like account names and authors exceeding 100 retrieved records in a day. The collector’s stricter sample contains {social["strict_sensitivity_records"]:,} records before the final text filters. Suspected automation is not proof of a bot. The main market signal uses comments, so a story-points threshold cannot directly filter it.'),
      paragraph(f'**Coverage limits.** The final news sample spans {s["news_sources"]:,} source labels after alias normalization. However, {news["google_rss_capped_requests"]} of {news["google_rss_requests"]} RSS requests reach 100 items: this is ranked, capped retrieval, not a media census. GDELT was rate-limited. X, Reddit and Bluesky supplied no usable corpus; HN represents a technology community, not all social-media users. Near-duplicate syndication and unidentified promotional sources can remain. English-oriented sources are supplemented by a detector-score sensitivity, not assumed to be perfectly English.'),
      paragraph(f'**Market timing.** RSS publication times often look normalized, so each RSS headline becomes usable at the next New York midnight. Guardian items use the later publication/modification clock; HN uses its timestamp. Each item enters the first 16:00 New York close at or after availability; weekend text enters Monday. Items whose conservative availability falls after the {end_label} close contribute only to descriptive comparisons. Text published after the final 16:00 cutoff is excluded. Latest wording is not a verified historical vintage; no live trading performance is claimed.'),
    ]
    third=[
      paragraph('Risk-topic detection and emotional wording are separate steps. Published regular expressions flag existential/control risk, misuse (including hacking and deception), and societal harms or safety. Positive mitigation discussions remain in the risk subset. VADER preserves negation, punctuation and capitalization; compound scores range from −1 to +1 and values at or below −0.05 are labeled negative. The Hu–Liu check divides negative-token counts by all alphabetic tokens, without stopword removal. Neither measure is an extinction probability or a validated attitude toward AI. [3–5]'),
      figure('sentiment_trends','Figure 2. Daily risk/safety wording with trailing averages for display only. Regressions use unsmoothed observations; no values are filled for missing days.','.94'),
      table(['Risk / safety sample','Sep 8–14 mean',f'Sep 22–{end.day} mean','Change'],[[m,number(trends.loc[m,'early_mean'],3),number(trends.loc[m,'late_mean'],3),number(trends.loc[m,'change'],3)] for m in ['News headlines','HN comments','HN stories']],['.36','.21','.21','.17']),
      paragraph(f'Calendar days receive equal weight in these opening/closing-period comparisons; the final day ends at 16:00. The news-negative share moves from {pct(trends.loc["News headlines","early_negative_share"])} to {pct(trends.loc["News headlines","late_negative_share"])}; comments move from {pct(trends.loc["HN comments","early_negative_share"])} to {pct(trends.loc["HN comments","late_negative_share"])}. Daily reversals and changes in retrieved composition prevent treating this as a monotonic public-opinion trend. {slope_result}'),
      paragraph('**Context audit.** An AI-assisted review of 60 real texts found quotation and target errors: a headline about scrutiny of OpenAI’s safety committee receives positive VADER valence, while a comment criticizing blame placed on AI receives negative valence. “Hacks” can receive zero valence. The class readings motivate comparing methods and inspecting these cases; their benchmark accuracies do not transfer to this corpus. No independent human accuracy estimate is available. [3–5]'),
    ]
    labels={'basket':'Basket','spread':'Basket − SPY','spread_qqq':'Basket − QQQ'}
    rows=[[labels[r.outcome],str(r.lag),str(r.n),number(r.effect_per_01,3),number(r.f,2),number(r.p,3),number(r.p_holm,3),number(r.wild_p,3)] for r in core.itertuples()]
    rev=rob.loc[rob.specification.str.startswith('reverse')].iloc[0]
    socialtest=rob.loc[rob.specification.eq('d_social_risk')].iloc[0]
    fourth=[
      paragraph(r'The source-balanced risk score is $S_t=(\bar{s}^{news}_t+\bar{s}^{comments}_t)/2$: equal source weights prevent comment volume from overwhelming news. A missing component makes the index missing. With $\Delta S_t=S_t-S_{t-1}$ and primary outcome $y_t=r^B_t-r^{SPY}_t$, estimate'),
      paragraph(r'$$y_t=\alpha+\sum_{j=1}^{p}\phi_j y_{t-j}+\sum_{j=1}^{p}\beta_j\Delta S_{t-j}+\varepsilon_t,\qquad H_0:\beta_1=\cdots=\beta_p=0.$$'),
      paragraph(f'Lag one is primary; lag two is a sensitivity. The F-test asks whether lagged sentiment adds predictive information beyond lagged returns. Lags are formed on the full session calendar before incomplete equations are deleted. Timing and missing observations leave **{int(main.n)} usable equations and {int(main.df_resid)} residual degrees of freedom** at lag one, and {int(second_lag.n)}/{int(second_lag.df_resid)} at lag two. Thousands of documents do not increase this time-series sample size. [9]'),
      table(['Outcome','Lags','N','0.1 × β₁','F','p','Holm p','Wild p'],rows,['.24','.07','.05','.14','.09','.10','.12','.12']),
      paragraph(f'Table 3. Effects are percentage points of return per 0.1-point larger prior sentiment change. At two lags, the coefficient shown is the first lag; F tests both jointly. Holm adjusts {len(core)} core F-tests together. Wild p-values are separate {int(main.bootstrap_draws):,}-draw restricted-residual fixed-design diagnostics; lagged regressors are held fixed. Conventional intervals and both p-value methods remain fragile in this sample.'),
      paragraph(f'The primary estimate is **{number(beta,3)} pp** per +0.1 change, with 95% interval **[{number(main.ci_low_per_01,3)}, {number(main.ci_high_per_01,3)}]**, raw p={number(main.p,3)}, Holm p={number(main.p_holm,3)} and wild p={number(main.wild_p,3)}. Its {"negative" if beta<0 else "positive"} sign means improved wording precedes {"weaker" if beta<0 else "stronger"} relative returns in this fit. This is a conditional association, not evidence that sentiment caused gains or losses. The smallest core adjusted p is {number(core.p_holm.min(),3)}.'),
      figure('influence','Figure 3. Primary coefficient after deleting one return equation, with all lag definitions preserved.','.93'),
      paragraph(f'Omitting {influence_label}, the most influential equation, changes the estimate to {number(influence.effect_per_01,3)} pp (p={number(influence.p,3)}). HN-only risk tone gives p={number(socialtest.p,3)}; reverse-direction testing gives p={number(rev.p,3)}. Source/author/thread weighting, stricter filtering, language restriction, extra delay and Hu–Liu checks appear in the notebook. {sensitivity_result} ADF and residual diagnostics are weak in such short samples. Even a Granger rejection would not identify structural causation.'),
    ]
    b=bias.loc['negative'];h=bias.loc['hu_pct']
    fifth=[
      paragraph('Sacerdote, Sehgal and Cook compare source groups, topics and audience popularity, using both dictionary negativity and a separately trained classifier. Their assigned November 2020 version finds 91% negative coverage among major US sources versus 54% among major non-US sources. Those figures are not VADER classification rates. Here, broad-AI collection supplies a comparison group, headlines provide a common news-text unit, and HN stories provide a limited popularity comparison. [2]'),
      figure('bias_comparison','Figure 4. Negative wording within each text type. Different text lengths and populations prevent treating the cross-medium gap as a causal media effect.','.98'),
      paragraph(f'Risk/safety headlines are {pct(desc.loc[("News headlines",True),"negative_share"])} negative versus {pct(desc.loc[("News headlines",False),"negative_share"])} for other AI headlines. Risk comments are {pct(desc.loc[("HN comments",True),"negative_share"])} negative. Hu–Liu negative words average {pct(desc.loc[("News headlines",True),"hu_negative"])} of tokens in risk headlines and {pct(desc.loc[("HN comments",True),"hu_negative"])} in risk comments. These are measurable wording differences, but risk keywords and headline compression mechanically contribute to them.'),
      paragraph(f'**Comparable publisher groups.** For {int(b.n):,} headlines from {int(b.clusters)} designated US/international publishers, a linear probability model controls publication-day effects, mutually exclusive risk-topic categories, log word count and its square. Standard errors cluster by publisher. US-group headlines are **{number(abs(100*b.coefficient),2)} percentage points {"less" if b.coefficient<0 else "more"} likely to be VADER-negative** (US-coefficient 95% interval {number(100*b.ci_low,2)} to {number(100*b.ci_high,2)} pp; p={number(b.p,3)}; two-measure Holm p={number(b.p_holm,3)}). The manually specified groups differ from the COVID paper’s outlet universe.'),
      paragraph(f'The Hu–Liu comparison is {number(h.coefficient,3)} pp in negative-token share (p={number(h.p,3)}; 95% interval {number(h.ci_low,3)} to {number(h.ci_high,3)} pp). {measure_conclusion} The VADER point estimate runs in the direction {bias_direction} the assigned study. Date/topic/length controls reduce observed composition differences but do not match every event or remove search-ranking selection.'),
      paragraph(f'**Attention selection.** HN stories with at least 5 snapshot points are {pct(eng.loc[True,"negative_share"])} negative versus {pct(eng.loc[False,"negative_share"])} below 5; risk-topic shares are {pct(eng.loc[True,"risk_share"])} versus {pct(eng.loc[False,"risk_share"])}. These are engagement-conditioned descriptions: age, exposure, topic and platform ranking also affect points. The comparison cannot establish reader demand or an ex-ante trading rule.'),
      paragraph('The defensible bias finding is a **sample-dependent asymmetry in wording and attention**. There is no observable daily ground-truth series for existential AI risk comparable to COVID case counts. The analysis cannot establish that journalists exaggerate risk, that negative reporting is false, or that the technology community represents the wider public.'),
    ]
    sixth=[
      paragraph(f'**Investment decision.** I would not trade this basket from the sign of the daily sentiment index. Its {number(s["returns"]["basket"])}% return differs from SPY by {s["returns"]["basket"]-s["returns"]["spy"]:+.2f} pp and QQQ by {s["returns"]["basket"]-s["returns"]["qqq"]:+.2f} pp. META returns {number(constituents.loc["META","cumulative_return_pct"])}%; the daily-rebalanced basket excluding META returns {number(exclusion.loc["META","return_pct"])}%. These comparisons show how stock composition affects the basket result. Benchmark subtraction is a relative return, not CAPM alpha or evidence that AI-risk news drove returns.'),
      paragraph('For an AI allocation, I would retain a diversified, separately risk-budgeted exposure and use risk coverage as a prompt for company-specific research: safety incidents, cloud demand, regulation, liability and investment spending. A high-reach warning can change the research agenda without supplying a statistically reliable buy/sell signal. I would require a longer out-of-sample window, contemporaneous text vintages, realistic trading costs and stable results across sources before permitting sentiment to change portfolio weights.'),
      paragraph('**Limits and reproducibility.** The study includes the September 30 close. Text ends at 16:00 New York time that day, so late-evening discussion is outside the sample. Short time series, capped retrieval, HN selection, imperfect risk rules, revisions and absent historical reach metrics prevent definitive causal or population-wide conclusions. The notebook exposes filters and model variants; cached source responses and hashes preserve the analyzed observations.'),
      paragraph(f'**Reproducibility.** The [repository]({REPO}) contains code, the executed notebook, dependencies, tests and an accurate AI-use statement. Raw texts, market data and generated data tables stay local and are excluded from Git. The PDF is a separate submission artifact. Codex provided substantial assistance; qualitative review is AI-assisted and does not substitute for independent student verification.'),
      paragraph('**References and data sources**'),
      paragraph('[1] Hubinger, E. (September 8, 2026). [X post](https://x.com/EvanHub/status/2097497037956891126). Assignment-provided event statement; no X corpus or verified reach metrics used.'),
      paragraph('[2] Sacerdote, B., Sehgal, R., and Cook, M. (2020). [Why Is All COVID-19 News Bad News?](https://www.nber.org/system/files/working_papers/w28110/w28110.pdf), NBER Working Paper 28110. Assigned November 2020 version.'),
      paragraph('[3] Carvalho, J., and Plastino, A. (2020). [On the evaluation and combination of state-of-the-art features in Twitter sentiment analysis](https://doi.org/10.1007/s10462-020-09895-6). Artificial Intelligence Review.'),
      paragraph('[4] Psomakelis, E., et al. (2014). [Comparing Methods for Twitter Sentiment Analysis](https://doi.org/10.5220/0005075302250232). KDIR, 225–232.'),
      paragraph('[5] Hutto, C. J., and Gilbert, E. (2014). [VADER](https://github.com/cjhutto/vaderSentiment), ICWSM; Hu, M., and Liu, B. (2004), [Mining and Summarizing Customer Reviews](https://doi.org/10.1145/1014052.1014073), KDD. Hu–Liu word lists from the archived NLTK corpus.'),
      paragraph('[6] [HN Algolia Search API](https://hn.algolia.com/api). Archived timestamp-filtered responses; [Hacker News API](https://github.com/HackerNews/API) for field definitions. [7] [Google News](https://news.google.com/) dated RSS search and [Guardian AI archive](https://www.theguardian.com/technology/artificialintelligenceai). Exact requests and publication clocks are in local manifests.'),
      paragraph('[8] [Yahoo Finance historical prices](https://finance.yahoo.com/quote/NVDA/history/), chart endpoints for the eight listed securities; archived adjusted and raw closes. [9] [statsmodels Granger documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html). Software version 0.14.4; independent tests check F-test equivalence.'),
    ]
    return [('1  Research question and equity basket',first),('2  Data collection, filtering and timing',second),
            ('3  Sentiment measurement and observed trend',third),('4  Does sentiment Granger-predict returns?',fourth),
            ('5  Is there measurable coverage bias?',fifth),('6  Investment interpretation and references',sixth)]
