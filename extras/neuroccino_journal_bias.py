import re, urllib.parse, pandas as pd, numpy as np
rows=[]
for line in open('neuroccino_raw.txt', encoding='utf-8'):
    p=line.rstrip('\n').split('\t')
    if len(p)<4 or not re.match(r'\d\d/\d\d/\d{4}',p[0]): continue
    d=pd.to_datetime(p[0],format='%d/%m/%Y')
    rows.append(dict(date=d,year=d.year,presenter=p[1].strip(),col3=p[2].strip(),url=urllib.parse.unquote(p[3].strip()),title=p[4].strip() if len(p)>4 else ''))
df=pd.DataFrame(rows)
M=[ # (substring, journal, family, kind)
 ('10.1038/s41583','Nature Reviews Neuroscience','Nature Portfolio','journal'),
 ('s41593','Nature Neuroscience','Nature Portfolio','journal'),
 ('s41586','Nature','Nature Portfolio','journal'),('d41586','Nature (news feature)','Nature Portfolio','journal'),
 ('s41467','Nature Communications','Nature Portfolio','journal'),
 ('s41562','Nature Human Behaviour','Nature Portfolio','journal'),
 ('s41592','Nature Methods','Nature Portfolio','journal'),
 ('s41928','Nature Electronics','Nature Portfolio','journal'),
 ('s41559','Nature Ecology & Evolution','Nature Portfolio','journal'),
 ('s42003','Communications Biology','Nature Portfolio','journal'),
 ('s41531','npj Parkinson\'s Disease','Nature Portfolio','journal'),
 ('s41598','Scientific Reports','Nature Portfolio','journal'),
 ('10.1126/sciadv','Science Advances','Science (AAAS)','journal'),
 ('10.1126/science','Science','Science (AAAS)','journal'),
 ('10.1073','PNAS','PNAS','journal'),
 ('journal.pbio','PLOS Biology','PLOS','journal'),('journal.pcbi','PLOS Computational Biology','PLOS','journal'),
 ('j.cortex','Cortex','Elsevier (other)','journal'),
 ('j.neuron','Neuron','Cell Press','journal'),('j.celrep','Cell Reports','Cell Press','journal'),
 ('j.tics','Trends in Cognitive Sciences','Cell Press','journal'),('j.cell.','Cell','Cell Press','journal'),
 ('S1474-4422','Lancet Neurology','Elsevier (other)','journal'),
 ('j.nbas','Aging Brain','Elsevier (other)','journal'),('j.dcn','Dev. Cogn. Neuroscience','Elsevier (other)','journal'),
 ('j.neubiorev','Neurosci. Biobehav. Rev.','Elsevier (other)','journal'),('j.biopsych','Biological Psychiatry','Elsevier (other)','journal'),
 ('10.1093/brain','Brain','Oxford UP','journal'),('10.1093/cercor','Cerebral Cortex','Oxford UP','journal'),
 ('imag_a','Imaging Neuroscience','MIT Press','journal'),('nol_a','Neurobiology of Language','MIT Press','journal'),
 ('eLife','eLife','eLife','journal'),
 ('10.1002/hbm','Human Brain Mapping','Wiley','journal'),('10.1111/joa','Journal of Anatomy','Wiley','journal'),
 ('s00429','Brain Structure & Function','Springer','journal'),('s00702','J. Neural Transmission','Springer','journal'),
 ('10.1523/ENEURO','eNeuro','SfN','journal'),
 ('10.1136/bmj','BMJ','BMJ','journal'),('10.1017/jme','J. Law, Medicine & Ethics','Cambridge UP','journal'),
 ('nejm.org','NEJM','NEJM','journal'),
 ('10.1177/00359157','Proc. Royal Soc. Medicine (historical)','Historical','historical'),
 ('biusante','Historical source','Historical','historical'),('mpi.nl','Historical source','Historical','historical'),
 ('10.1101','bioRxiv/medRxiv','Preprint','preprint'),('10.64898','bioRxiv/medRxiv (new DOI prefix)','Preprint','preprint'),
 ('arXiv','arXiv','Preprint','preprint'),
 ('nature.com/articles/s41593','Nature Neuroscience','Nature Portfolio','journal'),
 ('pubmed.ncbi','Unknown (PubMed link only)','Unknown','unknown'),
 ('neurodesk','Software tool (no paper)','None','none'),
]
def cls(u):
    for s,j,f,k in M:
        if s in u: return j,f,k
    return 'No paper/no DOI','None','none'
df[['journal','family','kind']]=df.url.apply(lambda u: pd.Series(cls(u)))
df.loc[df.url.isin(['','none','computer says NO']),['journal','family','kind']]=['No paper/no DOI','None','none']
df.to_csv('neuroccino_parsed.csv',index=False)
print(len(df),'sessions;', df.kind.value_counts().to_dict())
paper=df[df.kind.isin(['journal','preprint','historical','unknown'])]
print('papers',len(paper),'dups',paper.title.duplicated().sum())
print(paper.family.value_counts().to_string())
print('\nTop journals'); print(paper.journal.value_counts().head(12).to_string())
j=paper[paper.kind=='journal']
vc=j.journal.value_counts(); print('\npublished-journal items',len(j),'distinct journals',len(vc),'top5 share',round(vc.head(5).sum()/len(j),2),'HHI',round(((vc/len(j))**2).sum(),3))
flag={'Nature','Science','Cell','Neuron','Nature Neuroscience','Nature Reviews Neuroscience','Nature Human Behaviour','Nature Methods'}
j2=j.assign(flagship=j.journal.isin(flag))
print('flagship share of journal papers',round(j2.flagship.mean(),2),int(j2.flagship.sum()))
print('\nBY YEAR (share of papers)')
paper=paper.assign(nature=paper.family.eq('Nature Portfolio'),preprint=paper.kind.eq('preprint'),flagship=paper.journal.isin(flag))
print(paper.groupby('year').agg(n=('title','size'),nature=('nature','mean'),flagship=('flagship','mean'),preprint=('preprint','mean')).round(2).to_string())
print('\nBY col3 (second name)')
paper['col3g']=np.where(paper.col3.str.contains('Steph'),'Steph',paper.col3.replace('', 'blank'))
print(paper.col3g.value_counts().to_string())
paper['steph']=paper.col3g.eq('Steph')
print(paper.groupby('steph').agg(n=('title','size'),nature=('nature','mean'),flagship=('flagship','mean'),preprint=('preprint','mean')).round(2).to_string())
from scipy.stats import fisher_exact
for c in ('nature','flagship','preprint'):
    t=pd.crosstab(paper.steph,paper[c]); print(c,'Fisher p',round(fisher_exact(t.values)[1],3), t.values.tolist())
print('\nPRESENTER counts'); print(paper.presenter.value_counts().head(8).to_string())
lang=r'language|aphasi|speech|semantic|sentence|dyslexia|stutter'
wm=r'white matter|tract|diffusion|connectom|fascicle|myelin|fibers|axon'
llm=r'large language|language models|conversational diagnostic|reasoning models|foundation model|artificial intelligence|\bAI\b|machine'
for n,p in (('language',lang),('white matter/tractography',wm),('AI/LLM',llm)):
    m=paper.title.str.contains(p,case=False,regex=True)
    print(n,int(m.sum()),round(m.mean(),2))