"""
CNSeminar 10-Year Anniversary — Transcript Analysis Pipeline
============================================================
Produces publication-quality figures for Aperture Neuro special issue.

Expected folder structure:
    transcripts/
        2016/  talk_title.txt  ...
        2025/  ...

Usage:
    python cns_analysis_pipeline.py --folder /path/to/transcripts --output ./figures
    python cns_analysis_pipeline.py --folder /path/to/transcripts --topics 6 --track myelin fmri language
    python cns_analysis_pipeline.py --folder /path/to/transcripts --stratify

Arguments:
    --folder    Root folder with year subfolders                   [required]
    --output    Where to save figures (default: ./cns_figures)
    --topics    Number of LDA topics to extract (default: 6)
    --track     Keywords to track (space-separated, base forms)
    --start     First year to include (default: auto-detect)
    --end       Last year to include (default: auto-detect)
    --format    pdf | png | both (default: both)
    --dpi       PNG resolution (default: 300)
    --stratify  Also run per-content-type analysis and produce
                fig5_content_types and content_type_summary.csv
"""

import os, re, sys, csv, argparse, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')
_LEMMATIZER = WordNetLemmatizer()


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT-TYPE CLASSIFICATION
# ───────────────────────────────────────────────────────────────────────────────
# Derived from filename stems using naming conventions of the channel.
# Five broad categories used in the paper; granular labels kept internally.
# ═══════════════════════════════════════════════════════════════════════════════

# Broad category labels and display names (order = stacking order in fig5)
CONTENT_CATEGORIES = {
    'neuroccino':    'Neuroccino',
    'cns_seminar':   'CNS seminars',
    'course':        'Courses & educational series',
    'community':     'Community & open science',
    'other':         'Other / miscellaneous',
}

# Palette matched to broad categories — soft muted tones (redefined in STYLE block)
CATEGORY_PALETTE = {
    'neuroccino':  '#4A6FA5',   # muted slate-blue
    'cns_seminar': '#7A9E7E',   # muted sage-green
    'course':      '#B07D62',   # muted terracotta
    'community':   '#8B7BAD',   # muted violet
    'other':       '#AAAAAA',   # neutral grey
}


def classify_content_type(stem):
    """
    Map a transcript filename stem to a broad content category.
    Returns (broad_category, granular_label).
    """
    s = stem.lower()

    # ── Neuroccino (all variants) ─────────────────────────────────────────────
    if s in ('neurccino_live', 'neuriccino_live'):       # known typos kept
        return 'neuroccino', 'neuroccino_live'
    if 'neuroccino_live' in s:
        return 'neuroccino', 'neuroccino_live'
    if s.startswith('neuroccino') or 'neuroccino' in s:
        return 'neuroccino', 'neuroccino'

    # ── CNS seminars ──────────────────────────────────────────────────────────
    if (s.startswith('cnstalk') or '_cnstalk_' in s or
            s.startswith('dr_') or s.startswith('prof_') or
            s.startswith('assoc_prof_') or s.startswith('cnscomms') or
            s.startswith('cnsdebate') or s.startswith('cnsnightcap')):
        granular = ('cnscomms' if s.startswith('cnscomms') else
                    'cnsdebate' if s.startswith('cnsdebate') else
                    'cnsnightcap' if s.startswith('cnsnightcap') else 'cnstalk')
        return 'cns_seminar', granular

    # ── Courses & educational series ──────────────────────────────────────────
    if re.match(r'^\d+_', s):                            # numbered episodes
        return 'course', 'course_numbered'
    if 'workshop' in s or 'tutorial' in s or 'hands_on' in s or 'brainhack' in s:
        return 'course', 'workshop_tutorial'
    if 'phd_viva' in s or s.endswith('_viva'):
        return 'course', 'phd_viva'
    if 'undergraduate_lecture' in s or 'intro_101' in s:
        return 'course', 'course_numbered'

    # ── Community & open science ──────────────────────────────────────────────
    if ('bordeauxtea' in s or 'reproducibilitea' in s or
            s.startswith('march_rtea') or s.startswith('novembertea')):
        return 'community', 'reproducibilitea'
    if s.startswith('meettheeditor'):
        return 'community', 'meettheeditor'
    if s.startswith('histoccino'):
        return 'community', 'histoccino'
    if s.startswith('ohbm'):
        return 'community', 'ohbm'
    if 'paperpitch' in s or 'posterpitch' in s:
        return 'community', 'paperpitch'

    # ── Other / misc ──────────────────────────────────────────────────────────
    # Named-speaker talks without a cnstalk_ prefix (post-2022 convention shift)
    # and anything else unrecognised
    return 'other', 'other'


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOUND NEUROSCIENCE TERMS
# ───────────────────────────────────────────────────────────────────────────────
# Joined with underscores BEFORE tokenisation so they survive as single tokens.
# Longer phrases listed first to avoid partial matches.
# ═══════════════════════════════════════════════════════════════════════════════

COMPOUNDS = [
    # ── cortical anatomy ──────────────────────────────────────────────────────
    ('anterior cingulate cortex',           'anterior_cingulate_cortex'),
    ('posterior cingulate cortex',          'posterior_cingulate_cortex'),
    ('primary somatosensory cortex',        'primary_somatosensory_cortex'),
    ('primary motor cortex',               'primary_motor_cortex'),
    ('dorsolateral prefrontal cortex',      'dorsolateral_prefrontal_cortex'),
    ('dorsolateral prefrontal',             'dorsolateral_prefrontal'),
    ('ventromedial prefrontal cortex',      'ventromedial_prefrontal_cortex'),
    ('ventromedial prefrontal',             'ventromedial_prefrontal'),
    ('orbitofrontal cortex',                'orbitofrontal_cortex'),
    ('prefrontal cortex',                   'prefrontal_cortex'),
    ('visual cortex',                       'visual_cortex'),
    ('auditory cortex',                     'auditory_cortex'),
    ('motor cortex',                        'motor_cortex'),
    ('insular cortex',                      'insular_cortex'),
    ('entorhinal cortex',                   'entorhinal_cortex'),
    ('parahippocampal cortex',              'parahippocampal_cortex'),
    ('cingulate cortex',                    'cingulate_cortex'),
    ('inferior frontal gyrus',              'inferior_frontal_gyrus'),
    ('middle frontal gyrus',                'middle_frontal_gyrus'),
    ('superior frontal gyrus',              'superior_frontal_gyrus'),
    ('superior temporal sulcus',            'superior_temporal_sulcus'),
    ('superior temporal gyrus',             'superior_temporal_gyrus'),
    ('middle temporal gyrus',               'middle_temporal_gyrus'),
    ('inferior temporal gyrus',             'inferior_temporal_gyrus'),
    ('angular gyrus',                       'angular_gyrus'),
    ('supramarginal gyrus',                 'supramarginal_gyrus'),
    ('fusiform gyrus',                      'fusiform_gyrus'),
    ('parahippocampal gyrus',               'parahippocampal_gyrus'),
    ('postcentral gyrus',                   'postcentral_gyrus'),
    ('precentral gyrus',                    'precentral_gyrus'),
    ('temporal lobe',                       'temporal_lobe'),
    ('frontal lobe',                        'frontal_lobe'),
    ('parietal lobe',                       'parietal_lobe'),
    ('occipital lobe',                      'occipital_lobe'),
    ('limbic system',                       'limbic_system'),
    # ── white matter / tracts ─────────────────────────────────────────────────
    ('white matter',                        'white_matter'),
    ('grey matter',                         'grey_matter'),
    ('gray matter',                         'gray_matter'),
    ('corpus callosum',                     'corpus_callosum'),
    ('arcuate fasciculus',                  'arcuate_fasciculus'),
    ('uncinate fasciculus',                 'uncinate_fasciculus'),
    ('superior longitudinal fasciculus',    'superior_longitudinal_fasciculus'),
    ('inferior longitudinal fasciculus',    'inferior_longitudinal_fasciculus'),
    ('inferior fronto occipital fasciculus','inferior_fronto_occipital_fasciculus'),
    ('corticospinal tract',                 'corticospinal_tract'),
    ('corticobulbar tract',                 'corticobulbar_tract'),
    ('optic radiation',                     'optic_radiation'),
    ('cingulum bundle',                     'cingulum_bundle'),
    ('internal capsule',                    'internal_capsule'),
    ('external capsule',                    'external_capsule'),
    # ── subcortical ───────────────────────────────────────────────────────────
    ('basal ganglia',                       'basal_ganglia'),
    ('substantia nigra',                    'substantia_nigra'),
    ('ventral tegmental area',              'ventral_tegmental_area'),
    ('ventral tegmental',                   'ventral_tegmental'),
    ('locus coeruleus',                     'locus_coeruleus'),
    ('nucleus accumbens',                   'nucleus_accumbens'),
    ('globus pallidus',                     'globus_pallidus'),
    ('putamen caudate',                     'putamen_caudate'),
    ('thalamic nuclei',                     'thalamic_nuclei'),
    ('lateral geniculate',                  'lateral_geniculate'),
    ('medial geniculate',                   'medial_geniculate'),
    ('spinal cord',                         'spinal_cord'),
    ('brain stem',                          'brain_stem'),
    ('dorsal raphe',                        'dorsal_raphe'),
    # ── cell types ────────────────────────────────────────────────────────────
    ('mirror neuron',                       'mirror_neuron'),
    ('pyramidal neuron',                    'pyramidal_neuron'),
    ('purkinje cell',                       'purkinje_cell'),
    ('granule cell',                        'granule_cell'),
    ('interneuron',                         'interneuron'),
    ('astrocyte',                           'astrocyte'),
    ('oligodendrocyte',                     'oligodendrocyte'),
    ('microglia',                           'microglia'),
    ('glial cell',                          'glial_cell'),
    ('place cell',                          'place_cell'),
    ('grid cell',                           'grid_cell'),
    # ── cellular / physiological ──────────────────────────────────────────────
    ('action potential',                    'action_potential'),
    ('resting potential',                   'resting_potential'),
    ('membrane potential',                  'membrane_potential'),
    ('synaptic plasticity',                 'synaptic_plasticity'),
    ('long term potentiation',              'long_term_potentiation'),
    ('long term depression',                'long_term_depression'),
    ('spike timing',                        'spike_timing'),
    ('firing rate',                         'firing_rate'),
    ('axon diameter',                       'axon_diameter'),
    ('axon hillock',                        'axon_hillock'),
    ('node of ranvier',                     'node_of_ranvier'),
    ('conduction velocity',                 'conduction_velocity'),
    ('myelin sheath',                       'myelin_sheath'),
    ('g ratio',                             'g_ratio'),
    ('blood brain barrier',                 'blood_brain_barrier'),
    ('neurovascular coupling',              'neurovascular_coupling'),
    ('dendritic spine',                     'dendritic_spine'),
    ('synaptic vesicle',                    'synaptic_vesicle'),
    ('neurotransmitter release',            'neurotransmitter_release'),
    # ── neurotransmitters / receptors ─────────────────────────────────────────
    ('dopamine receptor',                   'dopamine_receptor'),
    ('serotonin receptor',                  'serotonin_receptor'),
    ('glutamate receptor',                  'glutamate_receptor'),
    ('gaba receptor',                       'gaba_receptor'),
    ('nmda receptor',                       'nmda_receptor'),
    ('ampa receptor',                       'ampa_receptor'),
    # ── cognitive / systems neuroscience ─────────────────────────────────────
    ('working memory',                      'working_memory'),
    ('long term memory',                    'long_term_memory'),
    ('short term memory',                   'short_term_memory'),
    ('episodic memory',                     'episodic_memory'),
    ('semantic memory',                     'semantic_memory'),
    ('spatial memory',                      'spatial_memory'),
    ('fear conditioning',                   'fear_conditioning'),
    ('extinction learning',                 'extinction_learning'),
    ('reward learning',                     'reward_learning'),
    ('reinforcement learning',              'reinforcement_learning'),
    ('theory of mind',                      'theory_of_mind'),
    ('self awareness',                      'self_awareness'),
    ('self consciousness',                  'self_consciousness'),
    ('perspective taking',                  'perspective_taking'),
    ('mental imagery',                      'mental_imagery'),
    ('visual imagery',                      'visual_imagery'),
    ('rubber hand illusion',                'rubber_hand_illusion'),
    ('rubber hand',                         'rubber_hand'),
    ('bodily self consciousness',           'bodily_self_consciousness'),
    ('predictive coding',                   'predictive_coding'),
    ('free energy principle',               'free_energy_principle'),
    ('bayesian brain',                      'bayesian_brain'),
    ('attention network',                   'attention_network'),
    ('executive function',                  'executive_function'),
    ('cognitive control',                   'cognitive_control'),
    ('response inhibition',                 'response_inhibition'),
    ('emotion regulation',                  'emotion_regulation'),
    ('social cognition',                    'social_cognition'),
    ('face processing',                     'face_processing'),
    ('speech perception',                   'speech_perception'),
    ('language production',                 'language_production'),
    ('language comprehension',              'language_comprehension'),
    ('visual word form',                    'visual_word_form'),
    ('phonological awareness',              'phonological_awareness'),
    ('phineas gage', 'phineas_gage'),
    # ── networks ──────────────────────────────────────────────────────────────
    ('default mode network',                'default_mode_network'),
    ('default mode',                        'default_mode'),
    ('salience network',                    'salience_network'),
    ('central executive network',           'central_executive_network'),
    ('frontoparietal network',              'frontoparietal_network'),
    ('dorsal attention network',            'dorsal_attention_network'),
    ('ventral attention network',           'ventral_attention_network'),
    ('resting state',                       'resting_state'),
    ('functional connectivity',             'functional_connectivity'),
    ('structural connectivity',             'structural_connectivity'),
    ('effective connectivity',              'effective_connectivity'),
    ('neural network',                      'neural_network'),
    ('large scale network',                 'large_scale_network'),
    # ── imaging / acquisition ─────────────────────────────────────────────────
    ('diffusion tensor imaging',            'diffusion_tensor_imaging'),
    ('diffusion weighted imaging',          'diffusion_weighted_imaging'),
    ('diffusion tensor',                    'diffusion_tensor'),
    ('magnetic resonance imaging',          'magnetic_resonance_imaging'),
    ('magnetic resonance spectroscopy',     'magnetic_resonance_spectroscopy'),
    ('magnetic resonance',                  'magnetic_resonance'),
    ('magnetic susceptibility',             'magnetic_susceptibility'),
    ('quantitative susceptibility mapping', 'quantitative_susceptibility_mapping'),
    ('quantitative susceptibility',         'quantitative_susceptibility'),
    ('source separation',                   'source_separation'),
    ('functional near infrared',            'functional_near_infrared'),
    ('positron emission tomography',        'positron_emission_tomography'),
    ('electroencephalography',              'electroencephalography'),
    ('magnetoencephalography',              'magnetoencephalography'),
    ('local field potential',               'local_field_potential'),
    ('blood oxygen level dependent',        'blood_oxygen_level_dependent'),
    ('blood oxygen level',                  'blood_oxygen_level'),
    ('voxel based morphometry',             'voxel_based_morphometry'),
    ('surface based analysis',              'surface_based_analysis'),
    ('independent component analysis',      'independent_component_analysis'),
    # ── stimulation / intervention ────────────────────────────────────────────
    ('transcranial magnetic stimulation',   'transcranial_magnetic_stimulation'),
    ('transcranial direct current',         'transcranial_direct_current'),
    ('deep brain stimulation',              'deep_brain_stimulation'),
    ('direct electrical stimulation',       'direct_electrical_stimulation'),
    ('electrocortical stimulation',         'electrocortical_stimulation'),
    ('cortical stimulation',                'cortical_stimulation'),
    ('optogenetic stimulation',             'optogenetic_stimulation'),
    # ── computational ─────────────────────────────────────────────────────────
    ('machine learning',                    'machine_learning'),
    ('deep learning',                       'deep_learning'),
    ('convolutional neural network',        'convolutional_neural_network'),
    ('convolutional neural',                'convolutional_neural'),
    ('recurrent neural network',            'recurrent_neural_network'),
    ('support vector machine',              'support_vector_machine'),
    ('support vector',                      'support_vector'),
    ('principal component analysis',        'principal_component_analysis'),
    ('principal component',                 'principal_component'),
    ('graph neural network',                'graph_neural_network'),
    ('generative model',                    'generative_model'),
    ('bayesian inference',                  'bayesian_inference'),
    ('monte carlo',                         'monte_carlo'),
    ('signal processing',                   'signal_processing'),
    # ── clinical / diseases ───────────────────────────────────────────────────
    ('multiple sclerosis',                  'multiple_sclerosis'),
    ('amyotrophic lateral sclerosis',       'amyotrophic_lateral_sclerosis'),
    ('parkinson disease',                   'parkinson_disease'),
    ('alzheimer disease',                   'alzheimer_disease'),
    ('traumatic brain injury',              'traumatic_brain_injury'),
    ('acquired brain injury',               'acquired_brain_injury'),
    ('brain tumour',                        'brain_tumour'),
    ('brain tumor',                         'brain_tumor'),
    ('glioblastoma multiforme',             'glioblastoma_multiforme'),
    ('awake craniotomy',                    'awake_craniotomy'),
    ('phantom limb',                        'phantom_limb'),
    ('chronic pain',                        'chronic_pain'),
    ('disorders of consciousness',          'disorders_of_consciousness'),
    ('minimally conscious state',           'minimally_conscious_state'),
    # ── open science / methods ────────────────────────────────────────────────
    ('open science',                        'open_science'),
    ('open access',                         'open_access'),
    ('reproducible research',               'reproducible_research'),
    ('preregistration',                     'preregistration'),
    ('effect size',                         'effect_size'),
    ('statistical power',                   'statistical_power'),
    ('false discovery rate',                'false_discovery_rate'),
    ('signal to noise',                     'signal_to_noise'),
    ('inter rater reliability',             'inter_rater_reliability'),
    # ── social / diversity ────────────────────────────────────────────────────
    ('virtual reality',                     'virtual_reality'),
    ('immersive virtual reality',           'immersive_virtual_reality'),
    ('social neuroscience',                 'social_neuroscience'),
    ('embodied cognition',                  'embodied_cognition'),
    ('racial bias',                         'racial_bias'),
    ('implicit bias',                       'implicit_bias'),
    ('low resource setting',                'low_resource_setting'),
    ('global south',                        'global_south'),
]


# ═══════════════════════════════════════════════════════════════════════════════
# STOPWORDS
# ───────────────────────────────────────────────────────────────────────────────
# Base forms only — lemmatisation handles inflections automatically.
# _GENERIC  = English function words + seminar chatter
# _NON_NEURO = words too vague to carry neuroscience signal
# ═══════════════════════════════════════════════════════════════════════════════

_GENERIC = """
a about above after again against all also am an and any are as at back be
because been before being below between both but by can cannot could did do
does doing down during each few for from further get go going got had has have
having he her here hers herself him himself his how if in into is it its itself
let me more most my myself no nor not now of off on once only or other our
ours ourselves out own per shall she should so some such than that the their
theirs them themselves then there these they this those through to too under
until up us was we were what when where which while who whom why will with
would you your yours yourself
yeah yep nope okay alright right sure absolutely exactly correct indeed
good great nice wonderful fantastic brilliant excellent amazing
thank thanks welcome hello goodbye bye everyone anybody somebody nobody
maybe perhaps probably hopefully guess suppose
sorry hear heard seeing look looked show showed shown showing
next previous forward move moved moving come came coming
start starting end ending finish finished beginning
continue continued following follow followed
work works worked working try tried trying
need needs want wants feel feels felt seem seemed seems
mean means meant include includes including
consider considers considering call calls called calling
mention mentioned discuss discussed discussing note noted notes
present presented presenting audience slide screen share sharing
video audio zoom teams online offline chat mute unmute microphone
connection internet streaming livestream host speaker attendee participant
session panel seminar conference workshop lecture meeting colleague
time minute hour today yesterday week month
quite really very think thinking thought went must
different didn wasn couldn wouldn shouldn wedi chimp
took take taken happen happened happening
remember remembered remembering
away already whether better
looking look wanted want doesn
give gives gave given find finds found
change changes changed changing
measure measures measured measuring
understand understands understood understanding
problem problems
normal normally
damage damages damaged
site sites
subject subjects
control controls controlled
test tests tested testing
word words
little pretty hard done hand
second third across related
question questions answer answers ask asked asking
something anything nothing everything
sort kind type bit lot
actually basically essentially generally typically obviously clearly simply
able put however although whereas
folder joining watching manuscript workflow container notebook
leah valentina eduardo steph bettencourt maruzzi mcgurk mcconaughey
figure talk exciting wondering
youtube streaming livestream
specialized bordeaux
morning install cloud folding homo sapiens lilith celsi
super higher link image adrian moruzzi hodgman magoon dove
script parcel particle
awesome setup sherrington piazza rapp kettering aviv
neurotino neurochino
ancora kawamura dewy informally accordion gaul azur
inanimate grapheme cydnabod file feature detail stuff discussion cool author layer
professor doctor immediately last color
phineas_gage phineas gage
especially sample presentation performance experiment parameter
lefel sachs gweledol semantig anodd cradling banking
lexicon hypoxia praxis musical tracheostomy
family mother believe brought fact tell lost sitting
neurosurgeon except discovered received nowadays
"""

_NON_NEURO = """
example case point part aspect term type kind form level
general specific lot many much less around within outside inside near
another else either neither entire particular
often always never sometimes usually frequently still even also
rather just only mainly mostly largely partly
course matter way approach finding
relevant similar various additional further possible likely certain
people well using same small first like take
year human quite response
study research paper work group
said saying told thing things know knowing knew knowledge
interesting interest field
something nothing everything anywhere somewhere
important importance significant significance
result results
information
number
difference
high low effect
use used
new make
based given

brain area region patient
took happened started remember away
might better whether already long
give find change measure understand problem
normal damage site subject control test word
little pretty hard done hand over across related
data looking analysis idea together wanted doesn
one two three four five six seven eight nine ten
eleven twelve twenty hundred thousand million first
main value science space side intact consequence verbal activity
gone happens eventually
method signal pattern task model individual neural
amelioration destructors initiative timbre fearful
empire enactment unsnipped setup
middle evidence immediately severe superior hole bone recall blood anxiety organ
"""

_NON_ENGLISH = """
gymdeithas golygfeydd aqueate pethau ddyn anifeiliaid ysgrifennol
csep ctbs synonymiadau ddarpariaeth tasc ysgolion obiect ddiffyg obiectau orthographic
camis subtest nonverbal eponym sickle spelling hypoxic crus anemia infarct
agnoesiaid golygfawr gair wnaeth bethau sensoriol dibynniad cyfrifoldebau brosesu 
cycloarchitecture broblem cradled cluck
interosseous convexity cydnabod
lefel gweledol semantig anodd tamping ysgrifennu tasg cyfeiriad provoke gwirioneddol geiriau gwneud
harlow broadband neanderthal iconic arbenning hytrach gwaith rhywbeth sylweddol
"""

_WELSH = """
mae gyda gydag yn yr yma yno chi nhw hwn hon hyn hynny fod fel
ond neu gan eto dim byd lle nid oes sut pam pryd beth pwy
feddwl llawr gweithgaredd anabod ffordd edrych nawr bach llawer
mwy llai hefyd felly wedyn achos oherwydd erbyn drwy trwy
dros dan rhwng ynghyd ynglyn ynghylch
agnoesiaid golygfawr brosesu cyfrifoldebau dibynniad sensoriol
gair wnaeth bethau feddwl ddarpariaeth synonymiau
gweithgaredd feddwl llawr anabod ffordd
florenz csap trident scusi unimanual adverb spelled descendant
ancora azur gaul cydnabod accordion dewy kawamura
"""

STOP = set(_GENERIC.split()) | set(_NON_NEURO.split()) | \
       set(_NON_ENGLISH.split()) | set(_WELSH.split())


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Greyscale palette for general figures (bars, lines, fills) ───────────────
# Colour is reserved for content-type and UMAP cluster differentiation only.
GREY_DARK   = '#2C2C2A'
GREY_MID    = '#888780'
GREY_LIGHT  = '#C8C7C2'
GREY_FILL   = '#E8E7E2'

# Single accent used sparingly for a highlighted series (e.g. words overlay)
ACCENT      = '#4A6FA5'   # muted slate-blue, CVD-safe

# General palette — greyscale sequence for multi-series plots (Fig3, Fig4 eras)
PALETTE = [
    '#2C2C2A',  # near-black
    '#5A5A58',  # dark grey
    '#888780',  # mid grey
    '#AAAAAA',  # light-mid grey
    '#C8C7C2',  # light grey
    '#4A6FA5',  # muted blue accent (only used when 5 series needed)
    '#7A9BBF',  # lighter blue accent
    '#BBBBBB',  # fallback grey
]

# ── Content-type palette — soft, muted, CVD-safe ─────────────────────────────
# Used ONLY in Fig5 (stacked bar) and Fig6 (category word clouds).
# Deliberately distinct from UMAP cluster colours below.
CATEGORY_PALETTE = {
    'neuroccino':  '#4A6FA5',   # muted slate-blue
    'cns_seminar': '#7A9E7E',   # muted sage-green
    'course':      '#B07D62',   # muted terracotta
    'community':   '#8B7BAD',   # muted violet
    'other':       '#AAAAAA',   # neutral grey
}

# ── UMAP cluster palette — soft, distinct from category palette ───────────────
# Used ONLY in cns_r2_embedding.py. Defined here for cross-script consistency.
CLUSTER_PALETTE = [
    '#D4956A',   # warm sand
    '#6A9FB5',   # dusty teal
    '#A8BF8C',   # sage
    '#C48BAD',   # dusty rose
    '#E3C97A',   # warm straw
    '#7EB5A6',   # seafoam
    '#B58DAE',   # lavender
    '#9BB5CC',   # steel blue
    '#D4A96A',   # pale amber
    '#88A87C',   # moss
]

CMAP_SEQUENTIAL = 'Greys'      # heatmaps: greyscale
CMAP_DIVERGING  = 'RdBu_r'     # diverging data
CMAP_HEATMAP    = 'Greys'      # TF-IDF and topic heatmaps

def set_style(dpi=300):
    plt.rcParams.update({
        'font.family': 'Lora',
        'font.size': 9,
        'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'figure.dpi': 150,
        'savefig.dpi': dpi, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08, 'legend.frameon': False,
        'legend.fontsize': 8,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(text):
    t = text.lower()
    t = re.sub(r"['\u2018\u2019\u201c\u201d\-]", ' ', t)
    t = re.sub(r'\s+', ' ', t)
    for phrase, joined in COMPOUNDS:
        t = t.replace(phrase, joined)
    return t


# Short neuroscience acronyms that must survive the length / vowel filters
# below (otherwise 'mri', 'eeg', 'pet', 'tms', ... are silently dropped).
KEEP_SHORT = {
    'mri', 'fmri', 'dmri', 'smri', 'eeg', 'meg', 'pet', 'spect', 'tms', 'tdcs',
    'dti', 'dwi', 'erp', 'ecog', 'seeg', 'nirs', 'fnirs', 'bold', 'hcp', 'gpu',
    'osf', 'sex', 'cnn', 'rnn', 'llm', 'fmri', 'dbs', 'ieeg', 'ica', 'pca',
}


def _keep_token(w):
    if w in STOP:
        return False
    if w in KEEP_SHORT:
        return True
    return (len(w) > 3
            and not all(c in 'bcdfghjklmnpqrstvwxyz' for c in w.replace('_', '')))


def tokenize(text):
    t = preprocess(text)
    tokens = re.findall(r'\b[a-z][a-z_]+[a-z]\b', t)
    lemmas = []
    for w in tokens:
        if '_' in w:
            lemmas.append(w)
        else:
            lemmas.append(_LEMMATIZER.lemmatize(w))
    return [w for w in lemmas if _keep_token(w)]


def display(token):
    return {'specie': 'species'}.get(token, token.replace('_', ' '))


# ═══════════════════════════════════════════════════════════════════════════════
# CORPUS LOADING
# ───────────────────────────────────────────────────────────────────────────────
# load_corpus        → tokens pooled by year (for main analysis)
# load_corpus_typed  → tokens pooled by year AND content type (for stratified)
# ═══════════════════════════════════════════════════════════════════════════════

def load_corpus(root_folder, start_year=None, end_year=None):
    root = Path(root_folder)
    years_data, file_counts = {}, {}
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        try:
            year = int(subfolder.name)
        except ValueError:
            continue
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue
        tokens, n = [], 0
        for fname in sorted(subfolder.glob('*.txt')):
            try:
                tokens.extend(tokenize(
                    fname.read_text(encoding='utf-8', errors='replace')))
                n += 1
            except Exception as e:
                print(f'  Warning: {fname.name}: {e}')
        if tokens:
            years_data[year], file_counts[year] = tokens, n
            print(f'  {year}: {n} talks, {len(tokens):,} tokens')
    return years_data, file_counts


def load_corpus_typed(root_folder, start_year=None, end_year=None):
    """
    Returns:
        years_data    — same as load_corpus (for reuse in main figs)
        file_counts   — same as load_corpus
        type_year_counts  — dict {broad_cat: {year: n_files}}
        type_year_tokens  — dict {broad_cat: {year: [tokens]}}
        file_manifest — list of dicts with per-file metadata
    """
    root = Path(root_folder)
    years_data, file_counts = {}, {}
    type_year_counts = defaultdict(lambda: defaultdict(int))
    type_year_tokens = defaultdict(lambda: defaultdict(list))
    file_manifest = []

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        try:
            year = int(subfolder.name)
        except ValueError:
            continue
        if start_year and year < start_year:
            continue
        if end_year and year > end_year:
            continue

        tokens_year, n = [], 0
        for fname in sorted(subfolder.glob('*.txt')):
            try:
                text = fname.read_text(encoding='utf-8', errors='replace')
                toks = tokenize(text)
                broad, granular = classify_content_type(fname.stem)
                tokens_year.extend(toks)
                type_year_counts[broad][year] += 1
                type_year_tokens[broad][year].extend(toks)
                n += 1
                file_manifest.append({
                    'year': year, 'stem': fname.stem,
                    'broad': broad, 'granular': granular,
                    'n_tokens': len(toks),
                })
            except Exception as e:
                print(f'  Warning: {fname.name}: {e}')

        if tokens_year:
            years_data[year] = tokens_year
            file_counts[year] = n
            print(f'  {year}: {n} talks, {len(tokens_year):,} tokens')

    return years_data, file_counts, type_year_counts, type_year_tokens, file_manifest


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_tfidf(years_data, years, max_features=500):
    corpora = [' '.join(years_data[yr]) for yr in years]
    vec = TfidfVectorizer(max_features=max_features, min_df=1,
                          stop_words=list(STOP))
    mat = vec.fit_transform(corpora)
    return mat, vec.get_feature_names_out()


def fit_lda(years_data, years, n_topics=6, max_features=500):
    corpora = [' '.join(years_data[yr]) for yr in years]
    vec = CountVectorizer(max_features=max_features, min_df=1,
                          stop_words=list(STOP))
    dtm = vec.fit_transform(corpora)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42,
        max_iter=150, learning_method='batch',
        doc_topic_prior=0.1, topic_word_prior=0.01)
    lda.fit(dtm)
    return lda, vec.get_feature_names_out(), lda.transform(dtm)


def freq_per_1k(tokens, word):
    key = _LEMMATIZER.lemmatize(word.lower().replace(' ', '_'))
    return Counter(tokens).get(key, 0) / max(len(tokens), 1) * 1000


def smooth(vals, w=1):
    return [np.mean(vals[max(0, i-w):min(len(vals), i+w+1)])
            for i in range(len(vals))]


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def save(fig, name, out, fmt):
    os.makedirs(out, exist_ok=True)
    if fmt in ('png', 'both'):
        fig.savefig(os.path.join(out, f'{name}.png'), dpi=300)
    if fmt in ('pdf', 'both'):
        fig.savefig(os.path.join(out, f'{name}.pdf'))
    plt.close(fig)
    print(f'  ✓  {name}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Channel activity overview + TF-IDF heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_overview_heatmap(years, file_counts, years_data, out, fmt):
    mat, terms = build_tfidf(years_data, years)
    var     = np.var(mat.toarray(), axis=0)
    top_idx = np.argsort(var)[-40:][::-1]
    h_terms = [display(terms[i]) for i in top_idx]
    h_data  = mat.toarray()[:, top_idx].T

    n_yrs  = len(years)
    fig_w  = min(16, max(11, 5 + n_yrs * 0.55))
    fig    = plt.figure(figsize=(fig_w, 6.5))
    gs     = gridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[1, max(1.8, n_yrs * 0.13)],
                               wspace=0.32)

    ax_a  = fig.add_subplot(gs[0])
    talks = [file_counts.get(yr, 0) for yr in years]
    words = [len(years_data[yr]) / 1000 for yr in years]
    ax_a.bar(years, talks, color=GREY_FILL, edgecolor='white',
             linewidth=0.4, zorder=2)
    ax_r = ax_a.twinx()
    ax_r.plot(years, words, color=ACCENT, lw=1.8, marker='o', ms=3.5, zorder=3)
    ax_r.spines['top'].set_visible(False)
    ax_r.tick_params(axis='y', labelcolor=ACCENT, labelsize=7.5)
    ax_r.set_ylabel('Total words (thousands)', color=ACCENT, fontsize=8)
    ax_a.set_xlabel('Year')
    ax_a.set_ylabel('Number of talks')
    ax_a.set_title('A   Channel activity over time', loc='left', pad=8)
    step = max(1, n_yrs // 8)
    ax_a.set_xticks(years[::step])
    ax_a.set_xticklabels([str(y) for y in years[::step]], rotation=45, ha='right')
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax_b = fig.add_subplot(gs[1])
    im   = ax_b.imshow(h_data, aspect='auto', cmap='Greys', interpolation='nearest')
    ax_b.set_xticks(range(n_yrs))
    ax_b.set_xticklabels([str(y)[2:] for y in years],
                          fontsize=max(6, 8 - n_yrs // 6))
    ax_b.set_yticks(range(len(h_terms)))
    ax_b.set_yticklabels(h_terms, fontsize=7.5)
    ax_b.set_xlabel('Year')
    ax_b.set_title('B   Term salience over time (TF-IDF)', loc='left', pad=8)
    for i in range(len(h_terms)+1): ax_b.axhline(i-0.5, color='white', lw=0.3)
    for j in range(n_yrs+1):        ax_b.axvline(j-0.5, color='white', lw=0.3)
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.02, pad=0.02)
    cbar.set_label('TF-IDF weight', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save(fig, 'fig1_activity_heatmap', out, fmt)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — LDA topic evolution
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_lda_topics(years, years_data, n_topics, out, fmt):
    lda, lda_terms, doc_topics = fit_lda(years_data, years, n_topics)
    topic_top    = [[display(lda_terms[j]) for j in c.argsort()[-5:]][::-1]
                    for c in lda.components_]
    short_labels = [f'T{i+1}: {" / ".join(w[:2])}'
                    for i, w in enumerate(topic_top)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8),
                             gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.08})

    ax = axes[0]
    ax.stackplot(np.array(years), doc_topics.T,
                 labels=short_labels, colors=PALETTE[:n_topics], alpha=0.75)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Year')
    ax.set_ylabel('Proportion of discourse')
    ax.set_title('A   Topic composition by year (LDA)', loc='left', pad=8)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8)

    ax2 = axes[1]
    ax2.set_title('B   Top words per topic', loc='left', pad=8)
    ax2.axis('off')
    row_h = 1.0 / (n_topics + 0.5)
    for t in range(n_topics):
        comp   = lda.components_[t]
        t_idx2 = comp.argsort()[-5:][::-1]
        words  = [display(lda_terms[j]) for j in t_idx2]
        wts    = comp[t_idx2] / comp[t_idx2].max()
        y0     = 1 - (t + 1) * row_h + 0.015
        ax2.add_patch(plt.Rectangle((0.01, y0 + row_h*0.3), 0.04, row_h*0.45,
                                    color=PALETTE[t], transform=ax2.transAxes,
                                    clip_on=False))
        ax2.text(0.07, y0 + row_h*0.52, f'Topic {t+1}',
                 transform=ax2.transAxes, fontsize=8, fontweight='bold',
                 color=PALETTE[t], va='center')
        for wi, (w, wt) in enumerate(zip(words, wts)):
            yw = y0 + row_h * (0.12 + wi * 0.14)
            ax2.barh(yw, wt*0.42, height=row_h*0.1, left=0.07,
                     color=PALETTE[t], alpha=0.7, transform=ax2.transAxes)
            ax2.text(0.07 + wt*0.42 + 0.01, yw, w,
                     transform=ax2.transAxes, fontsize=7, va='center', color='#2C2C2A')

    fig.subplots_adjust(bottom=0.25)
    save(fig, 'fig2_lda_topics', out, fmt)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Keyword frequency trends
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_keyword_trends(years, years_data, track_words, out, fmt):
    n    = len(track_words)
    ncol = min(5, n)
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 2.5, nrow * 2.6), sharey=False)
    flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, word in enumerate(track_words):
        ax    = flat[i]
        vals  = [freq_per_1k(years_data[yr], word) for yr in years]
        smo   = smooth(vals, w=1)
        ax.fill_between(years, smo, alpha=0.12, color=GREY_MID)
        ax.plot(years, smo, color=GREY_DARK, lw=1.8, zorder=3)
        ax.scatter(years, vals, color=GREY_DARK, s=15, zorder=4, alpha=0.55)
        ax.set_title(display(word), fontsize=9, fontweight='bold', pad=3)
        step = max(1, len(years) // 5)
        ax.set_xticks(years[::step])
        ax.set_xticklabels([str(y)[2:] for y in years[::step]], fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        if i % ncol == 0:
            ax.set_ylabel('Freq / 1k words', fontsize=7)

    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    y0, y1 = years[0], years[-1]
    fig.suptitle(f'Keyword frequency trends, {y0}–{y1} (per 1,000 words)',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    save(fig, 'fig3_keyword_trends', out, fmt)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Word clouds by era
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_wordclouds(years, years_data, out, fmt):
    n      = len(years)
    n_eras = min(4, n)
    ps     = max(1, n // n_eras)
    era_bounds = []
    for ei in range(n_eras):
        lo = ei * ps
        hi = (ei + 1) * ps if ei < n_eras - 1 else n
        era_bounds.append(years[lo:hi])
    # Greyscale shades per era — darkest = most recent
    era_greys = ['#2C2C2A', '#5A5A58', '#888780', '#AAAAAA']
    ERAS = [(yrs, f'{yrs[0]}–{yrs[-1]}' if len(yrs) > 1 else str(yrs[0]),
             era_greys[ei % len(era_greys)]) for ei, yrs in enumerate(era_bounds) if yrs]

    def cf_factory(hex_c):
        import matplotlib.colors as mc
        r, g, b = mc.to_rgb(hex_c)
        def cf(word, font_size, **kw):
            f = 0.3 + 0.7 * min(1, font_size / 80)
            return 'rgb(%d,%d,%d)' % (
                int((r+(1-r)*(1-f))*255),
                int((g+(1-g)*(1-f))*255),
                int((b+(1-b)*(1-f))*255))
        return cf

    fig, axes = plt.subplots(1, len(ERAS), figsize=(3.5 * len(ERAS), 3.8))
    if len(ERAS) == 1:
        axes = [axes]

    for ax, (era_yrs, label, col) in zip(axes, ERAS):
        tokens = [w for yr in era_yrs if yr in years_data
                  for w in years_data[yr] if w not in STOP]
        freq   = {display(w): c for w, c in Counter(tokens).items()}
        if not freq:
            ax.axis('off')
            continue
        wc = WordCloud(width=560, height=360, background_color='white',
                       max_words=60, max_font_size=90, min_font_size=11,
                       prefer_horizontal=0.85, color_func=cf_factory(col),
                       stopwords=set(), collocations=False)
        wc.generate_from_frequencies(freq)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(label, fontsize=10, fontweight='bold', pad=6)

    fig.suptitle('Vocabulary landscape by era', fontsize=11, fontweight='bold', y=1.03)
    plt.tight_layout()
    save(fig, 'fig4_wordclouds', out, fmt)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Content-type composition over time (stacked area + per-type bars)
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_content_types(years, file_counts, type_year_counts, out, fmt):
    """
    Panel A: stacked bar chart of video counts by content category per year.
    Panel B: horizontal bar chart of total counts per category (all years).
    """
    cats  = list(CONTENT_CATEGORIES.keys())
    labels = [CONTENT_CATEGORIES[c] for c in cats]
    colors = [CATEGORY_PALETTE[c] for c in cats]
    all_years = sorted(years)

    # Build matrix: rows = categories, cols = years
    mat = np.array([[type_year_counts[c].get(yr, 0) for yr in all_years]
                    for c in cats])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2),
                             gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.28})

    # ── Panel A: stacked bar by year ──────────────────────────────────────────
    ax = axes[0]
    bottoms = np.zeros(len(all_years))
    bar_w   = 0.72
    for i, (cat, col, lbl) in enumerate(zip(cats, colors, labels)):
        vals = mat[i]
        ax.bar(all_years, vals, bottom=bottoms, color=col, width=bar_w,
               label=lbl, edgecolor='white', linewidth=0.4, zorder=2)
        bottoms += vals

    # Grey band for the 2018-2019 gap
    ax.axvspan(2017.5, 2019.5, color='#f0efe8', zorder=1, linewidth=0)
    ax.text(2018.5, bottoms.max() * 0.5, 'no\ndata', ha='center', va='center',
            fontsize=7, color='#888780', style='italic')

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of videos')
    ax.set_title('A   Content-type composition by year', loc='left', pad=8)
    ax.set_xticks(all_years)
    ax.set_xticklabels([str(y) for y in all_years], rotation=45, ha='right', fontsize=7.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc='upper left', fontsize=7.5, ncol=1,
              bbox_to_anchor=(0.01, 0.99), borderpad=0.5)

    # ── Panel B: total per category ───────────────────────────────────────────
    ax2 = axes[1]
    totals = mat.sum(axis=1)
    # Sort by total descending for readability
    order  = np.argsort(totals)
    sorted_totals  = totals[order]
    sorted_labels  = [labels[i] for i in order]
    sorted_colors  = [colors[i] for i in order]

    bars = ax2.barh(range(len(cats)), sorted_totals, color=sorted_colors,
                    edgecolor='white', linewidth=0.4, height=0.6)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels(sorted_labels, fontsize=8)
    ax2.set_xlabel('Total videos (2010–2026)')
    ax2.set_title('B   Total by category', loc='left', pad=8)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Add count labels on bars
    for bar, val in zip(bars, sorted_totals):
        if val > 0:
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     str(int(val)), va='center', fontsize=8, color='#2C2C2A')

    save(fig, 'fig5_content_types', out, fmt)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Per-category word clouds (one per broad category)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_category_wordclouds(type_year_tokens, out, fmt):
    """One word cloud per broad content category."""
    cats   = list(CONTENT_CATEGORIES.keys())
    labels = [CONTENT_CATEGORIES[c] for c in cats]
    colors = [CATEGORY_PALETTE[c] for c in cats]

    def cf_factory(hex_c):
        import matplotlib.colors as mc
        r, g, b = mc.to_rgb(hex_c)
        def cf(word, font_size, **kw):
            f = 0.3 + 0.7 * min(1, font_size / 80)
            return 'rgb(%d,%d,%d)' % (
                int((r+(1-r)*(1-f))*255),
                int((g+(1-g)*(1-f))*255),
                int((b+(1-b)*(1-f))*255))
        return cf

    ncols = 3
    nrows = (len(cats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * 3.0))
    flat = axes.flatten() if len(cats) > 1 else [axes]

    for ax, cat, lbl, col in zip(flat, cats, labels, colors):
        # Pool all tokens across all years for this category
        tokens = [w for yr_toks in type_year_tokens[cat].values()
                  for w in yr_toks if w not in STOP]
        freq   = {display(w): c for w, c in Counter(tokens).items()}
        if not freq:
            ax.axis('off')
            ax.set_title(lbl, fontsize=9, fontweight='bold', pad=4)
            continue
        wc = WordCloud(width=500, height=320, background_color='white',
                       max_words=50, max_font_size=80, min_font_size=10,
                       prefer_horizontal=0.85, color_func=cf_factory(col),
                       stopwords=set(), collocations=False)
        wc.generate_from_frequencies(freq)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(lbl, fontsize=9, fontweight='bold', pad=4,
                     color=col)

    for j in range(len(cats), len(flat)):
        flat[j].set_visible(False)

    fig.suptitle('Vocabulary landscape by content category', fontsize=11,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    save(fig, 'fig6_category_wordclouds', out, fmt)


# ═══════════════════════════════════════════════════════════════════════════════
# CSV SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary_csv(years, type_year_counts, file_manifest, out):
    """
    Writes two CSV files:
      content_type_summary.csv  — cross-tab: rows=year, cols=category, + total
      file_manifest.csv         — one row per transcript with year/type/tokens
    """
    os.makedirs(out, exist_ok=True)
    cats      = list(CONTENT_CATEGORIES.keys())
    all_years = sorted(years)

    # Cross-tab
    summary_path = os.path.join(out, 'content_type_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['year'] + [CONTENT_CATEGORIES[c] for c in cats] + ['total']
        w.writerow(header)
        for yr in all_years:
            row = [yr] + [type_year_counts[c].get(yr, 0) for c in cats]
            row.append(sum(row[1:]))
            w.writerow(row)
        # Totals row
        tot_row = ['TOTAL']
        for c in cats:
            tot_row.append(sum(type_year_counts[c].values()))
        tot_row.append(sum(tot_row[1:]))
        w.writerow(tot_row)
    print(f'  ✓  content_type_summary.csv')

    # File manifest
    manifest_path = os.path.join(out, 'file_manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['year','stem','broad','granular','n_tokens'])
        w.writeheader()
        w.writerows(file_manifest)
    print(f'  ✓  file_manifest.csv')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TRACK = [
    'myelin', 'language', 'diversity', 'axon', 'machine_learning',
    'consciousness', 'white_matter', 'virtual_reality', 'decoding',
    'cortex', 'plasticity', 'network', 'tractography', 'working_memory',
    'deep_learning', 'open_science', 'connectivity', 'embodiment',
    'multiple_sclerosis', 'resting_state',
]


def main():
    p = argparse.ArgumentParser(description='CNSeminar transcript analysis')
    p.add_argument('--folder',   required=True)
    p.add_argument('--output',   default='./cns_figures')
    p.add_argument('--topics',   type=int, default=6)
    p.add_argument('--track',    nargs='+', default=DEFAULT_TRACK)
    p.add_argument('--start',    type=int, default=None)
    p.add_argument('--end',      type=int, default=None)
    p.add_argument('--format',   choices=['pdf', 'png', 'both'], default='both')
    p.add_argument('--dpi',      type=int, default=300)
    p.add_argument('--stratify', action='store_true',
                   help='Also produce content-type figures and CSV summary')
    args = p.parse_args()

    set_style(args.dpi)
    print(f'\nLoading transcripts from: {args.folder}')

    if args.stratify:
        years_data, file_counts, type_year_counts, type_year_tokens, file_manifest = \
            load_corpus_typed(args.folder, args.start, args.end)
    else:
        years_data, file_counts = load_corpus(args.folder, args.start, args.end)

    if not years_data:
        print('No data found. Check folder structure.')
        sys.exit(1)

    years = sorted(years_data.keys())
    total = sum(len(t) for t in years_data.values())
    print(f'\n{len(years)} years: {years[0]}–{years[-1]}  |  {total:,} tokens total')
    print(f'Saving to: {args.output}\n')

    print('[1/4] Activity overview + TF-IDF heatmap...')
    fig1_overview_heatmap(years, file_counts, years_data, args.output, args.format)

    print('[2/4] LDA topic evolution...')
    fig2_lda_topics(years, years_data, args.topics, args.output, args.format)

    print('[3/4] Keyword frequency trends...')
    fig3_keyword_trends(years, years_data, args.track, args.output, args.format)

    print('[4/4] Word clouds by era...')
    fig4_wordclouds(years, years_data, args.output, args.format)

    if args.stratify:
        print('\n[5/6] Content-type composition...')
        fig5_content_types(years, file_counts, type_year_counts, args.output, args.format)

        print('[6/6] Per-category word clouds...')
        fig6_category_wordclouds(type_year_tokens, args.output, args.format)

        print('\nWriting CSV summaries...')
        write_summary_csv(years, type_year_counts, file_manifest, args.output)

    print(f'\nAll outputs saved to {args.output}')


if __name__ == '__main__':
    main()