"""
theme.py

Design system partage par toutes les pages de l'application (variables CSS,
typographie, composants .card/.badge/.metric-card, etc.). Extrait de
l'ancien predict.py monolithique pour etre injecte une seule fois par
l'entree principale (predict.py) plutot que duplique dans chaque page.
"""

import streamlit as st


def inject_theme(rtl: bool = False, dark: bool = False) -> None:
    """
    Injecte le CSS global de l'application. A appeler une fois, avant
    st.navigation(). `rtl=True` (langue arabe) inverse la mise en page
    pour un rendu droite-a-gauche correct. `dark=True` applique la palette
    sombre (memes composants, memes classes CSS — seules les variables de
    couleur changent, donc aucune page n'a besoin d'etre modifiee pour
    supporter le mode sombre).
    """
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #F8FAFC;
        --bg-secondary: #F1F5F9;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F8FAFC;
        --bg-input: #F1F5F9;
        --border-color: #E2E8F0;
        --border-light: #CBD5E1;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --accent-primary: #2563EB;
        --accent-secondary: #60A5FA;
        --accent-gradient: linear-gradient(135deg, #2563EB, #1D4ED8);
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
        --warning-text: #92400E;
        /* Variantes texte/icone dediees (contraste WCAG AA reel, calcule -
           voir README section Accessibilite). Les tokens --success/
           --warning/--danger ci-dessus restent la palette de marque
           (#10B981/#F59E0B/#EF4444) pour les usages non textuels (fonds
           teintes de badge, graphiques) ; ces variantes -text, plus
           foncees, sont utilisees partout ou ces couleurs servent de texte
           ou d'icone lisible (badges, jauge de confiance, statuts du
           pipeline), car les tons de marque d'origine echouent le ratio
           4.5:1 (texte) et meme 3:1 (icones) sur fond clair. */
        --success-text: #047857;
        --danger-text: #B91C1C;
        --accent-text: #1D4ED8;
        --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-hover: 0 4px 12px rgba(37, 99, 235, 0.12);
        --radius: 12px;
        --radius-sm: 8px;
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: var(--text-primary) !important;
    }

    /* CORRECTION : retrait de !important sur span et label pour permettre
       aux icônes Material Symbols de Streamlit de s'afficher correctement
       (file uploader, expanders, etc.) */
    p, li, div {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-secondary);
    }

    span, label {
        font-family: 'Inter', sans-serif;
        color: var(--text-secondary);
    }

    .main-header {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: var(--shadow);
    }

    .main-header h1 {
        font-size: 30px;
        font-weight: 800 !important;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .main-header .subtitle {
        font-size: 15px;
        color: var(--text-secondary);
        font-weight: 400;
    }

    .readout-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px;
        background: var(--border-color);
        border-radius: var(--radius-sm);
        overflow: hidden;
        margin: 20px 0 28px 0;
    }

    .readout-item {
        background: var(--bg-card);
        padding: 14px 20px;
        transition: var(--transition);
    }

    .readout-item:hover {
        background: var(--bg-card-hover);
    }

    .readout-label {
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .readout-value {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        font-feature-settings: "tnum";
    }

    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 24px;
        transition: var(--transition);
        box-shadow: var(--shadow);
    }

    .card:hover {
        border-color: var(--accent-primary);
        box-shadow: var(--shadow-hover);
    }

    /* Reskin des conteneurs natifs st.container(border=True) pour qu'ils
       adoptent le meme style que .card. On les utilise a la place d'un
       <div class='card'> ouvert/ferme dans deux st.markdown() separes des
       qu'un composant natif (graphique, image, bouton...) doit se trouver
       a l'interieur : Streamlit ne "referme" pas un div HTML ouvert dans un
       st.markdown() precedent autour d'un composant natif suivant, ce qui
       laissait une boite vide a l'ecran. */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-card);
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius) !important;
        box-shadow: var(--shadow);
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--accent-primary) !important;
        box-shadow: var(--shadow-hover);
    }

    /* === BOUTONS MODERNES === */
    .stButton button, .stDownloadButton button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        transition: var(--transition) !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.20);
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35);
    }

    .stButton button:active, .stDownloadButton button:active {
        transform: scale(0.98);
    }

    /* Boutons d'export avec couleurs distinctes */
    .export-btn-primary {
        background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    }
    .export-btn-success {
        background: linear-gradient(135deg, #10B981, #059669) !important;
    }
    .export-btn-warning {
        background: linear-gradient(135deg, #F59E0B, #D97706) !important;
    }
    .export-btn-danger {
        background: linear-gradient(135deg, #EF4444, #DC2626) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-input) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: var(--radius) !important;
        transition: var(--transition) !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent-primary) !important;
        background: var(--bg-card) !important;
    }

    section[data-testid="stSidebar"] {
        background: var(--bg-card) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 24px 20px !important;
    }

    /* Navigation multi-pages (st.Page) : effet de survol doux et mise en
       evidence de la page active. Selecteurs defensifs — s'ils ne
       correspondent a rien dans une future version de Streamlit, ce bloc
       est un no-op silencieux plutot qu'une erreur. */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"],
    section[data-testid="stSidebar"] nav a {
        border-radius: var(--radius-sm);
        transition: var(--transition);
        margin-bottom: 2px;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover,
    section[data-testid="stSidebar"] nav a:hover {
        background: var(--bg-card-hover) !important;
        transform: translateX(2px);
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"],
    section[data-testid="stSidebar"] nav a[aria-current="page"] {
        background: rgba(37, 99, 235, 0.10) !important;
        border-left: 3px solid var(--accent-primary);
        font-weight: 600;
    }

    .sidebar-header {
        text-align: center;
        padding: 12px 0 20px 0;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 20px;
    }

    .sidebar-header h3 {
        font-size: 20px;
        font-weight: 700 !important;
        margin: 0;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar-header p {
        font-size: 12px;
        color: var(--text-muted);
        margin: 4px 0 0 0;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 12px;
        border-radius: var(--radius-sm);
        background: var(--bg-input);
        transition: var(--transition);
        margin-bottom: 4px;
    }

    .legend-item:hover {
        background: var(--bg-card-hover);
    }

    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .legend-text {
        font-size: 13px;
        font-weight: 500;
        color: var(--text-primary);
    }

    .legend-sub {
        font-size: 11px;
        color: var(--text-muted);
        margin-left: auto;
    }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        padding: 16px 20px;
        text-align: center;
        transition: var(--transition);
        box-shadow: var(--shadow);
    }

    .metric-card:hover {
        border-color: var(--accent-primary);
        box-shadow: var(--shadow-hover);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        font-feature-settings: "tnum";
    }

    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-top: 4px;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.12);
        color: var(--success-text);
    }

    .badge-warning {
        background: rgba(245, 158, 11, 0.12);
        color: var(--warning-text);
    }

    .badge-danger {
        background: rgba(239, 68, 68, 0.12);
        color: var(--danger-text);
    }

    .badge-info {
        background: rgba(37, 99, 235, 0.12);
        color: var(--accent-text);
    }

    .stProgress > div > div > div > div {
        background: var(--accent-gradient) !important;
        border-radius: 10px;
    }

    .stSlider [data-baseweb="slider"] {
        background: var(--bg-input) !important;
    }

    .stSlider [data-testid="stThumbValue"] {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        background: var(--bg-input) !important;
        border-color: var(--border-color) !important;
        border-radius: var(--radius-sm) !important;
    }

    .disclaimer {
        background: rgba(245, 158, 11, 0.06);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-radius: var(--radius-sm);
        padding: 12px 16px;
        font-size: 12px;
        color: var(--warning-text);
        margin-top: 16px;
    }

    .footer {
        text-align: center;
        padding: 28px 0 12px 0;
        border-top: 1px solid var(--border-color);
        margin-top: 40px;
    }

    .footer p {
        font-size: 13px;
        color: var(--text-muted);
        margin: 0;
    }

    .footer .highlight {
        color: var(--accent-primary);
        font-weight: 600;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.4s ease-out;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .pulse {
        animation: pulse 2s ease-in-out infinite;
    }

    .icon-spin {
        display: inline-flex;
        animation: iconSpin 1.2s linear infinite;
    }

    @keyframes iconSpin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Garde-fous responsives generiques : aucun element ne doit pouvoir
       forcer un defilement horizontal de la page sur tablette/mobile. */
    img, svg {
        max-width: 100%;
        height: auto;
    }

    .card, .metric-card, .main-header, .readout-strip {
        max-width: 100%;
        overflow-x: auto;
    }

    /* Etats de focus clavier (accessibilite). :focus-visible ne s'affiche
       qu'a la navigation clavier (Tab), pas au clic souris, conformement
       aux recommandations WCAG 2.4.7 (Focus Visible) sans etre intrusif
       pour la souris. */
    button:focus-visible,
    a:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    [tabindex]:focus-visible,
    [data-baseweb="select"]:focus-within,
    [data-testid="stSidebarNavLink"]:focus-visible {
        outline: 2px solid var(--accent-primary) !important;
        outline-offset: 2px !important;
        border-radius: 4px;
    }

    .stButton button:focus-visible,
    .stDownloadButton button:focus-visible {
        outline: 2px solid var(--accent-secondary) !important;
        outline-offset: 2px !important;
    }

    @media (max-width: 768px) {
        .main-header {
            padding: 20px;
        }
        .main-header h1 {
            font-size: 24px;
        }
        .readout-strip {
            grid-template-columns: 1fr 1fr;
        }
        .readout-item {
            padding: 12px 16px;
        }
        .readout-value {
            font-size: 16px;
        }
    }

    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }

    /* Personnalisation des sliders pour un meilleur rendu */
    [data-baseweb="slider"] div {
        background-color: var(--bg-input) !important;
    }
    [data-baseweb="slider"] div[role="slider"] {
        background-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.2) !important;
    }

    /* Ajustements responsives de base (tablette / ecran etroit). Non
       verifie visuellement (pas de navigateur dans l'environnement de
       developpement) — a valider sur un vrai appareil. */
    @media (max-width: 900px) {
        .card {
            padding: 16px !important;
        }
        .metric-value {
            font-size: 1.3rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    if rtl:
        st.markdown("""
        <style>
        /* Rendu droite-a-gauche (langue arabe). Cible le conteneur principal
           du contenu ; la sidebar et les composants Streamlit natifs
           (sliders, boutons) restent en LTR car leur inversion complete
           casserait leur interaction (curseurs, icones directionnelles). */
        .main .block-container {
            direction: rtl;
            text-align: right;
        }
        .main .block-container .card,
        .main .block-container .metric-card,
        .main .block-container .disclaimer {
            direction: rtl;
            text-align: right;
        }
        </style>
        """, unsafe_allow_html=True)

    if dark:
        st.markdown("""
        <style>
        /* Palette sombre : ne redeclare QUE les variables de couleur ; les
           regles CSS elles-memes (definies plus haut avec var(--xxx))
           s'appliquent automatiquement a toutes les pages sans qu'aucune
           page n'ait besoin d'etre modifiee pour supporter ce mode. */
        :root {
            --bg-primary: #0B1120;
            --bg-secondary: #131B2E;
            --bg-card: #161F36;
            --bg-card-hover: #1B2540;
            --bg-input: #1B2540;
            --border-color: #2A3655;
            --border-light: #3B4A73;
            --text-primary: #F1F5F9;
            --text-secondary: #CBD5E1;
            --text-muted: #7C89A6;
            --accent-primary: #3B82F6;
            --accent-secondary: #60A5FA;
            --accent-gradient: linear-gradient(135deg, #3B82F6, #2563EB);
            --warning-text: #FCD34D;
            --success-text: #34D399;
            --danger-text: #F87171;
            --accent-text: #93C5FD;
            --shadow: 0 1px 3px rgba(0,0,0,0.45), 0 1px 2px rgba(0,0,0,0.35);
            --shadow-hover: 0 4px 14px rgba(59, 130, 246, 0.25);
        }

        /* Certains composants natifs Streamlit (champs de saisie) ont leur
           propre fond code en dur ; on les force a suivre la palette sombre. */
        [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
            background-color: var(--bg-input) !important;
            color: var(--text-primary) !important;
        }
        </style>
        """, unsafe_allow_html=True)