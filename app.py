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

import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ─── Optional dependencies (graceful degradation) ─────────────
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

# ─── User-facing copy ─────────────────────────────────────────
INDICATOR_INFO = {
    'L': {
        'name': 'Words',
        'description': 'How loaded is the language?',
        'low_msg':  'Similar word choices',
        'mid_msg':  'Some loaded-word differences',
        'high_msg': 'Very different loaded vocabulary',
    },
    'E': {
        'name': 'People',
        'description': 'How are the same people described?',
        'low_msg':  'Same people described similarly',
        'mid_msg':  'Some differences in how people are described',
        'high_msg': 'Same people described very differently',
    },
    'C': {
        'name': 'Facts',
        'description': "What's covered vs left out?",
        'low_msg':  'Cover similar facts',
        'mid_msg':  'Some coverage differences',
        'high_msg': 'Cover very different facts',
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


# ════════════════════════════════════════════════════════════════════
#  INDICATOR COMPUTATION
# ════════════════════════════════════════════════════════════════════

def _tokenize(text):
    """Crude word tokenizer; fine for lexicon hit-counting."""
    return re.findall(r"\b[a-z][a-z'-]+\b", text.lower())


def compute_L(text_1, text_2, topic):
    """
    Words (L) — lexical framing divergence.
    For each article, compute the normalised share of frame_a vs frame_b
    lexicon hits. L = |article_1_lfi - article_2_lfi|.
    Returns L_raw + per-article hit spans for highlighting.
    """
    if topic not in LFI_LEXICONS:
        return None, [[], []], ('', '')
    lex = LFI_LEXICONS[topic]
    frame_a_name, frame_b_name = list(lex.keys())
    frame_a_words = {w.lower() for w in lex[frame_a_name]}
    frame_b_words = {w.lower() for w in lex[frame_b_name]}

    def article_lfi(text):
        tokens = _tokenize(text)
        n = max(len(tokens), 1)
        a = sum(1 for t in tokens if t in frame_a_words) / n
        b = sum(1 for t in tokens if t in frame_b_words) / n
        return a - b  # +ve toward frame_a

    L_raw = abs(article_lfi(text_1) - article_lfi(text_2))

    # Hit spans for highlighting (each: (start, end, surface, frame_name))
    def find_hits(text):
        hits = []
        all_words = list(frame_a_words) + list(frame_b_words)
        for w in all_words:
            pattern = r'\b' + re.escape(w) + r'\b'
            for m in re.finditer(pattern, text, re.IGNORECASE):
                frame = frame_a_name if w in frame_a_words else frame_b_name
                hits.append((m.start(), m.end(), m.group(), frame))
        # Deduplicate by span (some words may appear in both lexicons)
        seen = set()
        deduped = []
        for h in hits:
            key = (h[0], h[1])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)
        return deduped

    return L_raw, [find_hits(text_1), find_hits(text_2)], (frame_a_name, frame_b_name)


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
        doc = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
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
    deltas = {name: (e1[name], e2[name], abs(e1[name] - e2[name])) for name in shared}
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

    L_raw, L_hits, frame_names = compute_L(text_1, text_2, topic)
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
                'title': ('Unique to this article' if is_unique
                          else 'Mentioned in both articles'),
            }))
        return render_highlights(text, spans)

    if layer == 'People':
        ents = results['C']['ents_1'] if which_article == 1 else results['C']['ents_2']
        deltas = results['E']['deltas'] if results['E']['value'] is not None else {}
        if not deltas:
            return (html_escape(text).replace('\n', '<br>') +
                    '<br><em>(Sentiment-aware entity comparison unavailable — '
                    'see sidebar setup status.)</em>')
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
                'title': (f'Sentiment here: {s1:+.2f} · '
                          f'other article: {s2:+.2f} · Δ={delta:.2f}'),
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


def score_card(col, code, value, percentile):
    info = INDICATOR_INFO[code]
    if value is None:
        col.markdown(f'### {info["name"]}')
        col.warning('Not computed — see sidebar setup status')
        return

    if percentile is not None:
        if percentile < 33:
            msg, level = info['low_msg'], 'low'
        elif percentile < 66:
            msg, level = info['mid_msg'], 'mid'
        else:
            msg, level = info['high_msg'], 'high'
    else:
        msg, level = info['mid_msg'], 'mid'

    badge = {'low': '#27AE60', 'mid': '#F39C12', 'high': '#E74C3C'}[level]
    pct_html = (f'<div style="font-size:12px;color:#888;margin-top:6px">'
                f'Higher than {percentile:.0f}% of audited pairs</div>'
                if percentile is not None else '')

    col.markdown(f"""
        <div style="border:1px solid #E0E0E0;border-radius:8px;padding:14px;
                    background:#FAFAFA">
          <div style="font-size:13px;color:#666;margin-bottom:4px">
            {info['description']}
          </div>
          <div style="font-size:22px;font-weight:600;color:#222">
            {info['name']}
          </div>
          <div style="font-size:32px;font-weight:700;color:{badge};margin:6px 0">
            {value:.3f}
          </div>
          <div style="font-size:13px;color:{badge};font-weight:500">
            {msg}
          </div>
          {pct_html}
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


# ════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title('Framing Divergence Explorer')
    st.markdown('Compare how two news articles frame the same event using '
                'three interpretable indicators: **Words**, **People**, **Facts**.')

    st.subheader('Setup status')
    spacy_ok = SPACY_AVAILABLE and load_spacy() is not None
    google_ok = GOOGLE_AVAILABLE and load_google_client() is not None
    st.write(('✅ ' if spacy_ok else '⚠️ ') + 'spaCy (entity extraction)')
    st.write(('✅ ' if google_ok else '⚠️ ') + 'Google NLP (entity sentiment)')

    if not spacy_ok:
        st.info('Install spaCy: `pip install spacy` then '
                '`python -m spacy download en_core_web_sm`')
    if not google_ok:
        st.info('Set GOOGLE_APPLICATION_CREDENTIALS to enable the People '
                'indicator. Words and Facts work without it.')

    if Path(CORPUS_PATH).exists():
        st.write('✅ Corpus baseline loaded (percentile scoring enabled)')
    else:
        st.write('⚠️ `results_fdi_pairs.csv` not found — percentile scoring '
                 'disabled. Raw scores still shown.')

    st.subheader('About')
    st.caption('FDI Indicators framework · Masters thesis prototype · '
               'Monash University Faculty of IT.')


# ════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ════════════════════════════════════════════════════════════════════

st.title('Framing Divergence Explorer')
st.markdown('How differently do two news articles tell the same story? '
            'Provide two articles below to see the breakdown across three '
            'framing indicators.')

# ─── Step 1: input ─────────────────────────────────────────────
st.subheader('1. Provide two articles')

topic = st.selectbox(
    'Topic of the articles (selects the loaded-words lexicon)',
    options=list(LFI_LEXICONS.keys()),
    format_func=lambda t: t.replace('_', ' ').title(),
)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown('**Article 1**')
    outlet_1 = st.text_input('Outlet name', value='Outlet A', key='outlet_1')
    mode_1 = st.radio('Input method', ['Paste text', 'Upload .txt'],
                      key='mode_1', horizontal=True)
    if mode_1 == 'Paste text':
        text_1 = st.text_area('Article text', height=240, key='text_1',
                              placeholder='Paste the full article body here…')
    else:
        up_1 = st.file_uploader('Upload .txt', type=['txt'], key='up_1',
                                label_visibility='collapsed')
        text_1 = read_uploaded_file(up_1)

with col_r:
    st.markdown('**Article 2**')
    outlet_2 = st.text_input('Outlet name', value='Outlet B', key='outlet_2')
    mode_2 = st.radio('Input method', ['Paste text', 'Upload .txt'],
                      key='mode_2', horizontal=True)
    if mode_2 == 'Paste text':
        text_2 = st.text_area('Article text', height=240, key='text_2',
                              placeholder='Paste the full article body here…')
    else:
        up_2 = st.file_uploader('Upload .txt', type=['txt'], key='up_2',
                                label_visibility='collapsed')
        text_2 = read_uploaded_file(up_2)

if not (text_1.strip() and text_2.strip()):
    st.info('Provide text for both articles to see the comparison.')
    st.stop()

# Soft warning if articles look totally unrelated (very few shared entities)
# is computed implicitly via C below.

# ─── Compute indicators ────────────────────────────────────────
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

# Off-topic warning
if results['C']['value'] is not None and results['C']['value'] > 0.95:
    st.warning('These articles share very few entities — they may not be '
               'about the same event. Results will still be shown but '
               'interpret with care.')

# ─── Step 2: indicator cards ───────────────────────────────────
st.subheader('2. Indicator scores')
c1, c2, c3 = st.columns(3)
score_card(c1, 'L', results['L']['value'], percentiles['L'])
score_card(c2, 'E', results['E']['value'], percentiles['E'])
score_card(c3, 'C', results['C']['value'], percentiles['C'])

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
st.caption('Each axis is scaled to the 95th percentile of the audited corpus '
           '(or a fixed reference if the corpus is unavailable).')

# ─── Step 4: side-by-side with highlights ──────────────────────
st.subheader('4. Side-by-side comparison')

layer = st.radio('Highlight layer', ['Words', 'People', 'Facts'],
                 horizontal=True, key='layer')

legends = {
    'Words':  'Loaded vocabulary from each contrasting frame is highlighted; '
              'hover for the frame label.',
    'People': 'Entities mentioned in BOTH articles are highlighted in red; '
              'darker = larger sentiment difference between the two articles. '
              'Hover for the per-entity sentiment scores.',
    'Facts':  'Purple = entities unique to this article (covered here but '
              'not in the other). Grey = entities mentioned in both.',
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
