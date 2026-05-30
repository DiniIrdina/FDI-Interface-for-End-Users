"""
Framing Divergence Explorer — Streamlit application
====================================================
Compare how two news articles frame the same event, using three
interpretable indicators: Words (L), People (E), Facts (C).
 
Run with:
    streamlit run app.py
 
Setup:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
 
Optional: set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON
to enable the "People" (entity sentiment) indicator.
"""
 
import streamlit as st
 
# ─── Streamlit Cloud credentials handler ─────────────────────
# When deployed on Streamlit Community Cloud, read the Google
# service-account JSON from st.secrets and write it to a temp file
# so the Google client library can find it. No-op when run locally
# (st.secrets won't contain 'gcp_service_account' on a laptop).
import os
import json
import tempfile
if 'gcp_service_account' in st.secrets:
    sa_path = os.path.join(tempfile.gettempdir(), 'sa.json')
    with open(sa_path, 'w') as f:
        json.dump(dict(st.secrets['gcp_service_account']), f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_path
 
# ─── Other imports ───────────────────────────────────────────
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
 
# ─── Optional dependencies (graceful degradation) ────────────
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
 
try:
    from google.cloud import language_v1
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
 
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
 
 
# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════
 
st.set_page_config(
    page_title='Framing Divergence Explorer',
    page_icon='📰',
    layout='wide',
    initial_sidebar_state='expanded',
)
 
CORPUS_PATH = 'results_fdi_pairs.csv'
ARTICLES_PATH = 'articles.csv'

# ─── LFI lexicons ─────────────────────────────────────────────
LFI_LEXICONS = {
    'immigration': {
        'humanitarian': [
            'refugee', 'asylum seeker', 'asylum', 'displaced', 'flee',
            'persecution', 'human rights', 'humanitarian', 'families',
            'children', 'vulnerable', 'protection', 'shelter', 'aid',
            'crisis', 'desperate', 'seeking safety', 'undocumented',
            'migrant workers', 'dream', 'opportunity'
        ],
        'security': [
            'illegal', 'illegals', 'alien', 'invasion', 'border security',
            'enforcement', 'deportation', 'deport', 'criminal', 'gang',
            'smuggling', 'trafficking', 'threat', 'surge', 'flood',
            'overwhelm', 'overrun', 'national security', 'crackdown',
            'raid', 'detain', 'arrest', 'trespass', 'lawbreaker'
        ]
    },
    'environment': {
        'concern': [
            'crisis', 'emergency', 'catastrophe', 'disaster', 'extinction',
            'irreversible', 'tipping point', 'devastating', 'urgent',
            'scientists warn', 'unprecedented', 'accelerating', 'threat',
            'carbon emissions', 'fossil fuels', 'greenhouse', 'pollution',
            'warming', 'rising sea levels', 'extreme weather', 'biodiversity',
            'ecosystem', 'sustainability', 'renewable', 'clean energy'
        ],
        'skeptical': [
            'alarmist', 'hoax', 'natural cycle', 'exaggerated', 'costly',
            'regulation', 'economic impact', 'job losses', 'burden',
            'overreach', 'uncertain', 'debate', 'models', 'prediction',
            'agenda', 'ideology', 'taxpayer'
        ]
    },
    'political conflict': {
        'people-centric': [
            'protesters', 'demonstrators', 'activists', 'peaceful',
            'movement', 'grassroots', 'civil rights', 'freedom fighters',
            'liberation', 'resistance', 'solidarity', 'uprising',
            'civilians', 'victims', 'displaced', 'suffering',
            'human rights', 'oppression', 'persecution', 'atrocities',
            'massacre', 'brutality', 'injustice', 'freedom',
            'self-determination', 'occupied', 'besieged'
        ],
        'state-centric': [
            'terrorists', 'insurgents', 'militants', 'extremists',
            'rioters', 'agitators', 'separatists', 'radicals',
            'aggression', 'provocation', 'destabilise', 'subversion',
            'national security', 'sovereignty', 'territorial integrity',
            'law and order', 'counterterrorism', 'crackdown',
            'retaliation', 'deterrence', 'regime change', 'threat',
            'hostile', 'illegal', 'incitement', 'propaganda'
        ]
    }
}

# ─── User-facing copy (Flesch-Kincaid Grade ≤ 10) ─────────────
INDICATOR_INFO = {
    'L': {
        'name': 'Words',
        'description': 'How loaded is the language?',
    },
    'E': {
        'name': 'People',
        'description': 'How are the same people described?',
    },
    'C': {
        'name': 'Facts',
        'description': "What's covered vs left out?",
    },
}
 
 
# ════════════════════════════════════════════════════════════════════
#  RESOURCE LOADERS  (cached so they load once per session)
# ════════════════════════════════════════════════════════════════════
 
@st.cache_resource
def load_spacy():
    if not SPACY_AVAILABLE:
        return None
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        return None
 
@st.cache_resource
def load_google_client():
    if not GOOGLE_AVAILABLE:
        return None
    try:
        return language_v1.LanguageServiceClient()
    except Exception:
        return None
 
@st.cache_data
def load_corpus_distributions():
    if not Path(CORPUS_PATH).exists():
        return None
    pairs = pd.read_csv(CORPUS_PATH)
    return {
        'L': pairs['L_raw'].dropna().values,
        'E': pairs['E_raw'].dropna().values,
        'C': pairs['C_raw'].dropna().values,
    }
 
@st.cache_data
def load_corpus_articles():
    """Returns DataFrame of audited articles, or None if not present."""
    if not Path(ARTICLES_PATH).exists():
        return None
    df = pd.read_csv(ARTICLES_PATH)
    # Be flexible about column names
    for col in ('content', 'text', 'body', 'article_text'):
        if col in df.columns:
            df = df.rename(columns={col: 'content'})
            break
    return df
 
 
# ════════════════════════════════════════════════════════════════════
#  INDICATOR COMPUTATION
# ════════════════════════════════════════════════════════════════════
 
def compute_L(text_1, text_2, topic, nlp=None):
    """
    Words (L) — lexical framing divergence with lemma-based matching.
    Matches the lfi_score_lemma operationalisation in api_pipeline_2.ipynb.
    Falls back to exact matching if spaCy is unavailable.
    """
    if topic not in LFI_LEXICONS:
        return None, [[], []], ('', '')
    lex = LFI_LEXICONS[topic]
    frame_a_name, frame_b_name = list(lex.keys())
 
    # ── Build the lemma (or exact) sets from the lexicons ─────
    if nlp is not None:
        def lemmatise_lex(words):
            lemmas = set()
            for w in words:
                doc = nlp(w.lower())
                for tok in doc:
                    if tok.is_alpha:
                        lemmas.add(tok.lemma_.lower())
            return lemmas
        frame_a_set = lemmatise_lex(lex[frame_a_name])
        frame_b_set = lemmatise_lex(lex[frame_b_name])
    else:
        frame_a_set = {w.lower() for w in lex[frame_a_name]}
        frame_b_set = {w.lower() for w in lex[frame_b_name]}
 
    # ── Process each article once: count + collect hit spans ──
    def process(text):
        if nlp is not None:
            doc = nlp(text)
            tokens = [tok for tok in doc if tok.is_alpha]
            n = max(len(tokens), 1)
            a_count = b_count = 0
            hits = []
            for tok in tokens:
                key = tok.lemma_.lower()
                if key in frame_a_set:
                    a_count += 1
                    hits.append((tok.idx, tok.idx + len(tok.text),
                                 tok.text, frame_a_name))
                elif key in frame_b_set:
                    b_count += 1
                    hits.append((tok.idx, tok.idx + len(tok.text),
                                 tok.text, frame_b_name))
            return (a_count - b_count) / n, hits
        else:
            tokens = re.findall(r"\b[a-z][a-z'-]+\b", text.lower())
            n = max(len(tokens), 1)
            a = sum(1 for t in tokens if t in frame_a_set) / n
            b = sum(1 for t in tokens if t in frame_b_set) / n
            hits = []
            for w in list(frame_a_set) + list(frame_b_set):
                for m in re.finditer(r'\b' + re.escape(w) + r'\b',
                                     text, re.IGNORECASE):
                    frame = frame_a_name if w in frame_a_set else frame_b_name
                    hits.append((m.start(), m.end(), m.group(), frame))
            return a - b, hits
 
    lfi_1, hits_1 = process(text_1)
    lfi_2, hits_2 = process(text_2)
    return abs(lfi_1 - lfi_2), [hits_1, hits_2], (frame_a_name, frame_b_name)
 
 
def _extract_entities_spacy(text, nlp, types=('PERSON', 'GPE', 'ORG', 'LOC')):
    """Return (start, end, text, label) tuples for entities of interest."""
    if nlp is None:
        return []
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.text, ent.label_)
            for ent in doc.ents if ent.label_ in types]
 
 
def compute_C(text_1, text_2, nlp):
    """
    Facts (C) — coverage divergence via entity-set Jaccard distance.
    C_raw = 1 - |A ∩ B| / |A ∪ B|.
    Returns C_raw + per-article entity lists + unique-entity sets.
    """
    if nlp is None:
        return None, [], [], set(), set()
 
    ents_1 = _extract_entities_spacy(text_1, nlp)
    ents_2 = _extract_entities_spacy(text_2, nlp)
 
    set_1 = {e[2].lower().strip() for e in ents_1}
    set_2 = {e[2].lower().strip() for e in ents_2}
    union = set_1 | set_2
    if not union:
        return 0.0, ents_1, ents_2, set(), set()
 
    jaccard = len(set_1 & set_2) / len(union)
    return 1.0 - jaccard, ents_1, ents_2, set_1 - set_2, set_2 - set_1
 
 
def compute_E(text_1, text_2, client):
    """
    People (E) — entity treatment divergence.
    Uses Google NLP entity-sentiment scores; for entities in BOTH articles,
    computes mean |Δ sentiment|. Returns E_raw + per-entity deltas.
    """
    if client is None:
        return None, {}
 
    def analyze(text):
        doc = language_v1.Document(content=text,
                                   type_=language_v1.Document.Type.PLAIN_TEXT)
        resp = client.analyze_entity_sentiment(request={'document': doc})
        return {e.name.lower().strip(): e.sentiment.score for e in resp.entities}
 
    try:
        e1 = analyze(text_1)
        e2 = analyze(text_2)
    except Exception as exc:
        st.warning(f'Google NLP API error: {exc}')
        return None, {}
 
    shared = set(e1) & set(e2)
    if not shared:
        return 0.0, {}
    deltas = {name: (e1[name], e2[name], abs(e1[name] - e2[name]))
              for name in shared}
    E_raw = float(np.mean([d[2] for d in deltas.values()]))
    return E_raw, deltas
 
 
@st.cache_data(show_spinner='Analysing articles…')
def compute_all_indicators(text_1, text_2, topic, _nlp_marker, _google_marker):
    """
    Returns a dict {'L': {...}, 'E': {...}, 'C': {...}}.
    The underscore arguments are present so the cache invalidates when
    backend availability changes; their values are not used.
    """
    nlp = load_spacy()
    client = load_google_client()
 
    L_raw, L_hits, frame_names = compute_L(text_1, text_2, topic, nlp)
    C_raw, ents_1, ents_2, uniq_1, uniq_2 = compute_C(text_1, text_2, nlp)
    E_raw, E_deltas = compute_E(text_1, text_2, client)
 
    return {
        'L': {'value': L_raw, 'hits': L_hits, 'frames': frame_names},
        'E': {'value': E_raw, 'deltas': E_deltas},
        'C': {'value': C_raw, 'ents_1': ents_1, 'ents_2': ents_2,
              'unique_1': uniq_1, 'unique_2': uniq_2},
    }
 
 
# ════════════════════════════════════════════════════════════════════
#  HIGHLIGHTING  (returns HTML strings for st.markdown)
# ════════════════════════════════════════════════════════════════════
 
def html_escape(text):
    return (text.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;').replace('"', '&quot;'))
 
 
def render_highlights(text, spans):
    """spans: list of (start, end, html_attrs_dict). Non-overlapping; sorted."""
    if not spans:
        return html_escape(text).replace('\n', '<br>')
    spans = sorted(spans, key=lambda s: s[0])
    parts, cursor = [], 0
    for start, end, attrs in spans:
        if start < cursor:
            continue  # skip overlapping spans
        parts.append(html_escape(text[cursor:start]))
        style = ';'.join(f'{k}:{v}' for k, v in attrs.items()
                         if k not in ('title',))
        title = attrs.get('title', '')
        parts.append(
            f'<mark style="{style};padding:1px 3px;border-radius:3px" '
            f'title="{html_escape(title)}">{html_escape(text[start:end])}</mark>'
        )
        cursor = end
    parts.append(html_escape(text[cursor:]))
    return ''.join(parts).replace('\n', '<br>')
 
 
def highlight_for_layer(text, layer, results, which_article):
    """which_article: 1 or 2. Returns an HTML string."""
    if layer == 'Words':
        hits = results['L']['hits'][which_article - 1]
        frame_a, frame_b = results['L']['frames']
        spans = []
        for start, end, word, frame in hits:
            colour = '#E67E22' if frame == frame_a else '#16A085'
            spans.append((start, end, {
                'background-color': f'{colour}33',
                'border-bottom': f'2px solid {colour}',
                'title': frame,
            }))
        return render_highlights(text, spans)
 
    if layer == 'Facts':
        ents = results['C']['ents_1'] if which_article == 1 else results['C']['ents_2']
        unique = (results['C']['unique_1'] if which_article == 1
                  else results['C']['unique_2'])
        spans = []
        for start, end, ent_text, _label in ents:
            is_unique = ent_text.lower().strip() in unique
            colour = '#9B59B6' if is_unique else '#BDC3C7'
            spans.append((start, end, {
                'background-color': f'{colour}33',
                'border-bottom': f'2px solid {colour}',
                'title': ('Only in this article' if is_unique
                          else 'Also in the other article'),
            }))
        return render_highlights(text, spans)
 
    if layer == 'People':
        ents = results['C']['ents_1'] if which_article == 1 else results['C']['ents_2']
        deltas = results['E']['deltas'] if results['E']['value'] is not None else {}
        if not deltas:
            return (html_escape(text).replace('\n', '<br>') +
                    '<br><em>(People comparison is not available yet — '
                    'see sidebar.)</em>')
        spans = []
        for start, end, ent_text, _label in ents:
            key = ent_text.lower().strip()
            if key not in deltas:
                continue
            s1, s2, delta = deltas[key]
            alpha = min(0.15 + delta * 0.5, 0.9)
            spans.append((start, end, {
                'background-color': f'rgba(231,76,60,{alpha:.2f})',
                'border-bottom': '2px solid #E74C3C',
                'title': (f'In this article: {s1:+.2f} · '
                          f'In the other: {s2:+.2f} · Difference: {delta:.2f}'),
            }))
        return render_highlights(text, spans)
 
    return html_escape(text).replace('\n', '<br>')
 
 
# ════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ════════════════════════════════════════════════════════════════════
 
def percentile_of(value, dist):
    if value is None or dist is None or len(dist) == 0 or not SCIPY_AVAILABLE:
        return None
    return float(stats.percentileofscore(dist, value, kind='mean'))
 
 
def headline_finding(results):
    """One-sentence summary picking out the most divergent indicator."""
    scored = {}
    for code, max_v in [('L', 2.0), ('E', 0.3), ('C', 1.0)]:
        v = results[code]['value']
        if v is not None:
            scored[code] = v / max_v
    if not scored:
        return ''
    top = max(scored, key=scored.get)
    if scored[top] < 0.2:
        return ('These two articles tell the story in much the same way. '
                'The differences below are small.')
    msg = {
        'L': "These two articles use very different words to describe the "
             "same event.",
        'E': "These two articles describe the same people quite differently.",
        'C': "These two articles cover very different facts, even though "
             "they're about the same event.",
    }[top]
    return msg
 
 
def score_card(col, code, value, percentile, headline_top_code=None):
    """Card with a plain-English sentence built from the indicator value."""
    info = INDICATOR_INFO[code]
    if value is None:
        col.markdown(f'### {info["name"]}')
        col.warning('Not available yet — see sidebar.')
        return
 
    # Plain-English sentence using the actual numbers
    if code == 'L':
        pct = int(round(value * 100))
        sentence = (f'About <b>{pct}%</b> of the loaded words in these '
                    f'articles are different.')
    elif code == 'E':
        if percentile is not None and percentile >= 66:
            sentence = (f'The way these articles describe shared people '
                        f'differs more than <b>{int(percentile)}%</b> of '
                        f'article pairs we studied. (Average sentiment '
                        f'gap: <b>{value:.2f}</b> on a −1 to +1 scale.)')
        else:
            sentence = (f'Shared people are described with sentiment scores '
                        f'<b>{value:.2f}</b> points apart on average '
                        f'(scale: −1 to +1).')
    elif code == 'C':
        shared_pct = int(round((1 - value) * 100))
        diff_pct = 100 - shared_pct
        sentence = (f'Out of all the people and places mentioned across '
                    f'both articles, <b>{shared_pct}%</b> appear in both. '
                    f'The other <b>{diff_pct}%</b> are only in one or '
                    f'the other.')
    else:
        sentence = ''
 
    # Pick a colour band: use percentile if we have it, else absolute thresholds
    if percentile is not None:
        if percentile < 33:
            level = 'low'
        elif percentile < 66:
            level = 'mid'
        else:
            level = 'high'
        pct_phrase = (f'More different than <b>{int(percentile)}%</b> of '
                      f'article pairs we studied.')
    else:
        level = 'mid'
        pct_phrase = ''
 
    headline_word = {'low': 'Articles agree',
                     'mid': 'Some difference',
                     'high': 'Very different'}[level]
    badge = {'low': '#27AE60', 'mid': '#F39C12', 'high': '#E74C3C'}[level]
    star = ' ★' if code == headline_top_code else ''
 
    col.markdown(f"""
        <div style="border:1px solid #E0E0E0;border-radius:8px;padding:14px;
                    background:#FAFAFA;min-height:200px">
          <div style="font-size:12px;color:#888;text-transform:uppercase;
                      letter-spacing:0.5px">{info['name']}{star}</div>
          <div style="font-size:13px;color:#666;margin-bottom:8px">
            {info['description']}
          </div>
          <div style="font-size:20px;font-weight:700;color:{badge};
                      margin:8px 0">{headline_word}</div>
          <div style="font-size:13px;color:#333;line-height:1.5">
            {sentence}
          </div>
          <div style="font-size:11px;color:#999;margin-top:8px;font-style:italic">
            {pct_phrase}
          </div>
        </div>
    """, unsafe_allow_html=True)
 
 
def radar_plot(scores_and_max):
    labels, plot_vals = [], []
    for code in ('L', 'E', 'C'):
        info = INDICATOR_INFO[code]
        v, mx = scores_and_max.get(code, (None, None))
        labels.append(info['name'])
        if v is None or mx is None or mx == 0:
            plot_vals.append(0)
        else:
            plot_vals.append(min(v / mx, 1.0))
    labels.append(labels[0])
    plot_vals.append(plot_vals[0])
 
    fig = go.Figure(go.Scatterpolar(
        r=plot_vals, theta=labels, fill='toself',
        line=dict(color='#27AE60', width=2),
        fillcolor='rgba(39,174,96,0.25)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                   showticklabels=False)),
        showlegend=False, height=320,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig
 
 
def read_uploaded_file(uploaded):
    if uploaded is None:
        return ''
    try:
        return uploaded.read().decode('utf-8')
    except UnicodeDecodeError:
        return uploaded.read().decode('latin-1')
 
 
def article_input(label_key, articles_df):
    """
    Render input controls for one article. Returns (text, outlet, topic_hint).
    `topic_hint` is the article's topic when picked from the corpus,
    otherwise None.
    """
    modes = ['Paste text', 'Upload .txt file']
    if articles_df is not None:
        modes.insert(0, 'Choose from our corpus')
 
    mode = st.radio('Input method', modes, key=f'mode_{label_key}',
                    horizontal=True)
 
    if mode == 'Choose from our corpus':
        topics = sorted(articles_df['topic'].dropna().unique())
        sel_topic = st.selectbox('Topic', topics, key=f'corpus_topic_{label_key}')
        topic_df = articles_df[articles_df['topic'] == sel_topic]
 
        # Use event_id if available, otherwise group by something we have
        if 'event_id' in topic_df.columns:
            events = sorted(topic_df['event_id'].dropna().unique())
            sel_event = st.selectbox('Event', events,
                                     key=f'corpus_event_{label_key}')
            event_df = topic_df[topic_df['event_id'] == sel_event]
        else:
            event_df = topic_df
 
        # Build picker options with a short preview to disambiguate
        options = {}
        for _, row in event_df.iterrows():
            preview = str(row.get('content', ''))[:80].replace('\n', ' ')
            label = f"{row.get('outlet', 'Unknown')} — {preview}…"
            options[label] = (str(row.get('content', '')),
                              str(row.get('outlet', f'Outlet {label_key}')))
        if not options:
            st.warning('No articles found for that topic and event.')
            return '', f'Outlet {label_key}', sel_topic
 
        sel = st.selectbox('Article', list(options.keys()),
                           key=f'corpus_article_{label_key}')
        text, outlet = options[sel]
        return text, outlet, sel_topic
 
    elif mode == 'Paste text':
        outlet = st.text_input('Outlet name', value=f'Outlet {label_key}',
                               key=f'outlet_{label_key}')
        text = st.text_area('Article text', height=240,
                            key=f'text_{label_key}',
                            placeholder='Paste the full article body here…')
        return text, outlet, None
 
    else:  # Upload .txt file
        outlet = st.text_input('Outlet name', value=f'Outlet {label_key}',
                               key=f'outlet_{label_key}')
        up = st.file_uploader('Upload', type=['txt'], key=f'up_{label_key}',
                              label_visibility='collapsed')
        return read_uploaded_file(up), outlet, None
 
 
# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════
 
with st.sidebar:
    st.title('Framing Divergence Explorer')
    st.markdown('Compare two news articles about the same event. '
                'We score each pair on three things: **Words**, **People**, '
                'and **Facts**.')
 
    st.subheader('Setup status')
    spacy_ok = SPACY_AVAILABLE and load_spacy() is not None
    google_ok = GOOGLE_AVAILABLE and load_google_client() is not None
    st.write(('✅ ' if spacy_ok else '⚠️ ') + 'spaCy (finds people and places)')
    st.write(('✅ ' if google_ok else '⚠️ ') + 'Google NLP (scores how people '
             'are described)')
 
    if not spacy_ok:
        st.info('Install spaCy: `pip install spacy` then '
                '`python -m spacy download en_core_web_sm`')
    if not google_ok:
        st.info('Set GOOGLE_APPLICATION_CREDENTIALS to turn on the People '
                'score. Words and Facts work without it.')
 
    if Path(CORPUS_PATH).exists():
        st.write('✅ Reference data loaded')
    else:
        st.write('⚠️ Reference data not found. Raw scores still shown.')
 
    if Path(ARTICLES_PATH).exists():
        st.write('✅ Audited corpus loaded')
    else:
        st.write('⚠️ Audited corpus not found. Paste/upload still works.')
 
    st.subheader('About')
    st.caption('FDI Indicators framework · Masters thesis prototype · '
               'Monash University.')
 
 
# ════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ════════════════════════════════════════════════════════════════════
 
st.title('Framing Divergence Explorer')
st.markdown('How differently do two news articles tell the same story? '
            'Add two articles below to compare them on three measures.')
 
# ─── How to read this (expander) ──────────────────────────────
with st.expander('ℹ️ How to read these scores'):
    st.markdown("""
    Each score looks at a different way two articles can differ:
 
    - **Words** — Do the articles use different *loaded* words for the
      same thing? One article might say "freedom fighter" where another
      says "rebel" — same person, different word, different feeling.
    - **People** — When both articles mention the same person, do they
      describe that person in similar or different ways?
    - **Facts** — Do the articles cover the same facts and people, or
      did each one focus on different parts of the story?
 
    Two articles can score high on one and low on another. That's the
    whole point. The shape on the radar plot below shows which kind of
    difference matters most for your pair.
    """)
 
 
# ─── Step 1: input ────────────────────────────────────────────
st.subheader('1. Provide two articles')
 
corpus_articles = load_corpus_articles()
 
col_l, col_r = st.columns(2)
with col_l:
    st.markdown('**Article 1**')
    text_1, outlet_1, topic_hint_1 = article_input('A', corpus_articles)
with col_r:
    st.markdown('**Article 2**')
    text_2, outlet_2, topic_hint_2 = article_input('B', corpus_articles)
 
# Topic for the LFI lexicon. Pre-select the hint if both corpus picks agree.
default_topic = None
if topic_hint_1 and topic_hint_2 and topic_hint_1 == topic_hint_2:
    default_topic = topic_hint_1
elif topic_hint_1 or topic_hint_2:
    default_topic = topic_hint_1 or topic_hint_2
 
available_topics = list(LFI_LEXICONS.keys())
topic_index = (available_topics.index(default_topic)
               if default_topic in available_topics else 0)
 
topic = st.selectbox(
    'Topic of the articles (picks the loaded-words list)',
    options=available_topics,
    index=topic_index,
    format_func=lambda t: t.replace('_', ' ').title(),
)
 
if not (text_1.strip() and text_2.strip()):
    st.info('Provide text for both articles to see the comparison.')
    st.stop()
 
 
# ─── Compute indicators ───────────────────────────────────────
nlp_marker = 'on' if load_spacy() else 'off'
google_marker = 'on' if load_google_client() else 'off'
results = compute_all_indicators(text_1, text_2, topic, nlp_marker, google_marker)
 
corpus = load_corpus_distributions()
percentiles = {}
for code in ('L', 'E', 'C'):
    v = results[code]['value']
    if corpus is not None and v is not None:
        percentiles[code] = percentile_of(v, corpus[code])
    else:
        percentiles[code] = None
 
# Soft warning when articles look unrelated
if results['C']['value'] is not None and results['C']['value'] > 0.95:
    st.warning('These articles share very few people or places — they may '
               'not be about the same event. Results will still be shown.')
 
 
# ─── Headline finding ─────────────────────────────────────────
headline_text = headline_finding(results)
if headline_text:
    st.markdown(
        f'<div style="background:#EAF6FF;border-left:4px solid #3498DB;'
        f'padding:14px 18px;border-radius:6px;font-size:16px;'
        f'color:#1A3A5C;margin-top:10px">'
        f'<b>Headline finding</b><br>{headline_text}</div>',
        unsafe_allow_html=True,
    )
 
# Which indicator is "most different" — used to star the relevant card
def _top_code(results):
    scored = {}
    for code, max_v in [('L', 2.0), ('E', 0.3), ('C', 1.0)]:
        v = results[code]['value']
        if v is not None:
            scored[code] = v / max_v
    return max(scored, key=scored.get) if scored else None
 
top_code = _top_code(results)
 
 
# ─── Step 2: indicator cards ──────────────────────────────────
st.subheader('2. Indicator scores')
c1, c2, c3 = st.columns(3)
score_card(c1, 'L', results['L']['value'], percentiles['L'], top_code)
score_card(c2, 'E', results['E']['value'], percentiles['E'], top_code)
score_card(c3, 'C', results['C']['value'], percentiles['C'], top_code)
 
# Entity counts: makes the Facts score concrete by showing how many
# unique people / places / organisations each article mentions.
if results['C']['value'] is not None:
    ents_1_set = {e[2].lower().strip() for e in results['C']['ents_1']}
    ents_2_set = {e[2].lower().strip() for e in results['C']['ents_2']}
    n1 = len(ents_1_set)
    n2 = len(ents_2_set)
    shared = len(ents_1_set & ents_2_set)
    st.caption(
        f'**{outlet_1}** mentions {n1} distinct people and places · '
        f'**{outlet_2}** mentions {n2} · Only **{shared}** are in both.'
    )
 
# A note about the People scale, shown only when E is in the red band
if percentiles['E'] is not None and percentiles['E'] >= 66:
    st.caption(
        '💡 The People scale is sensitive — most news articles describe '
        'shared people very consistently, so even small numbers (like '
        '0.05) can be unusual. We compare your pair against the article '
        'pairs we studied to decide what counts as a large difference.'
    )
 
 
# ─── Step 3: radar profile ────────────────────────────────────
st.subheader('3. Profile')
maxes = {}
for code in ('L', 'E', 'C'):
    if corpus is not None:
        maxes[code] = float(np.percentile(corpus[code], 95))
    else:
        maxes[code] = {'L': 2.0, 'E': 0.3, 'C': 1.0}[code]
sam = {code: (results[code]['value'], maxes[code]) for code in ('L', 'E', 'C')}
st.plotly_chart(radar_plot(sam), use_container_width=True)
st.caption('Each line is scaled against the most divergent article pairs '
           'we studied. The closer a point is to the outer edge, the more '
           'this pair stands out.')
 
 
# ─── Step 4: side-by-side with highlights ─────────────────────
st.subheader('4. Side-by-side comparison')
 
layer = st.radio('Highlight layer', ['Words', 'People', 'Facts'],
                 horizontal=True, key='layer')
 
legends = {
    'Words':  ('Loaded words from each side are highlighted. Hover over a '
               'word to see which side it belongs to.'),
    'People': ('People mentioned in both articles are highlighted in red. '
               'The darker the red, the more the two articles differ in '
               'how they describe that person. Hover to see the scores.'),
    'Facts':  ('Purple shows people or places that appear in this article '
               'but not in the other. Grey shows ones that appear in both.'),
}
st.caption(legends[layer])
 
col_l2, col_r2 = st.columns(2)
with col_l2:
    st.markdown(f'**{outlet_1}**')
    st.markdown(
        f'<div style="border:1px solid #E0E0E0;border-radius:6px;'
        f'padding:12px;background:#FFFFFF;line-height:1.7;'
        f'max-height:600px;overflow-y:auto">'
        f'{highlight_for_layer(text_1, layer, results, 1)}</div>',
        unsafe_allow_html=True,
    )
with col_r2:
    st.markdown(f'**{outlet_2}**')
    st.markdown(
        f'<div style="border:1px solid #E0E0E0;border-radius:6px;'
        f'padding:12px;background:#FFFFFF;line-height:1.7;'
        f'max-height:600px;overflow-y:auto">'
        f'{highlight_for_layer(text_2, layer, results, 2)}</div>',
        unsafe_allow_html=True,
    )
 
 
# ─── Footer ────────────────────────────────────────────────────
st.markdown('---')
st.caption('Framing Divergence Explorer · proof-of-concept prototype · '
           'Built on the FDI Indicators framework (Ubaidah, 2026, Masters '
           'thesis, Monash University).')
