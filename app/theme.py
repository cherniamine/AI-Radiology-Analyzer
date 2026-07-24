"""
theme.py

Design system partage par toutes les pages de l'application (variables CSS,
typographie, composants .card/.badge/.metric-card, etc.). Extrait de
l'ancien predict.py monolithique pour etre injecte une seule fois par
l'entree principale (predict.py) plutot que duplique dans chaque page.

Identite visuelle : "negatoscope" — la boite lumineuse sur laquelle un
radiologue lit un film argentique. Le mode sombre (identite premiere) est
une salle de lecture presque noire, chaleur ambree qui rappelle la lumiere
diffusee par le lightbox ; le mode clair est la meme salle, lumieres
allumees. Les deux partagent la meme famille de tons chauds (jamais de
bleu-froid neutre) pour rester reconnaissables l'un de l'autre.

Toutes les valeurs de couleur texte/icone (`--*-text`) ont ete verifiees
programme en main (voir tests/test_accessibility_contrast.py) pour
respecter WCAG 2.1 AA — 4.5:1 pour le texte, 3:1 pour les icones/elements
graphiques — a la fois sur fond de carte plein et sur fond de badge teinte.
"""

import streamlit as st


def inject_theme(rtl: bool = False, dark: bool = False) -> None:
    """
    Injecte le CSS global de l'application. A appeler une fois, avant
    st.navigation(). `rtl=True` (langue arabe) inverse la mise en page
    pour un rendu droite-a-gauche correct. `dark=True` applique la palette
    "negatoscope" (memes composants, memes classes CSS — seules les
    variables de couleur changent, donc aucune page n'a besoin d'etre
    modifiee pour supporter le mode sombre).
    """

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        /* Salle de lecture, lumieres allumees — meme famille chaude que le
           mode sombre (jamais de gris/bleu neutre) pour rester coherent. */
        --bg-primary: #F7F3EC;
        --bg-secondary: #F0E9DC;
        --bg-card: #FFFFFF;
        --bg-card-hover: #FBF7EF;
        --bg-input: #F3EDE1;
        --border-color: #E5DCC9;
        --border-light: #D8CBB0;
        --text-primary: #241C10;
        --text-secondary: #5E5240;
        --text-muted: #726550;
        --accent-primary: #C2790C;
        --accent-secondary: #E8A23D;
        --accent-gradient: linear-gradient(135deg, #E8A23D, #C2790C);
        --on-accent: #1A1005;
        /* Tons de marque (fonds teintes de badge, graphiques, remplissages
           non textuels) — pas utilises directement comme couleur de texte. */
        --success: #3FB27F;
        --warning: #E8B93E;
        --danger: #E5484D;
        /* Variantes texte/icone dediees (contraste WCAG AA reel, calcule -
           voir README section Accessibilite). Les tokens --success/
           --warning/--danger ci-dessus restent la palette de marque pour
           les usages non textuels (fonds teintes de badge, graphiques) ;
           ces variantes -text, plus foncees, sont utilisees partout ou ces
           couleurs servent de texte ou d'icone lisible (badges, jauge de
           confiance, statuts du pipeline), car les tons de marque d'origine
           echouent le ratio 4.5:1 (texte) et meme 3:1 (icones) sur fond
           clair. */
        --success-text: #0B6942;
        --warning-text: #7A4E00;
        --danger-text: #9E2226;
        --accent-text: #9C5000;
        --shadow: 0 1px 3px rgba(36,28,16,0.08), 0 1px 2px rgba(36,28,16,0.05);
        --shadow-hover: 0 6px 20px rgba(194, 121, 12, 0.18);
        --shadow-glow: 0 0 0 1px rgba(194, 121, 12, 0.25);
        --radius: 12px;
        --radius-sm: 8px;
        --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        --font-display: 'Space Grotesk', 'Inter', sans-serif;
        --font-body: 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', 'Consolas', monospace;
    }

    .stApp {
        background: var(--bg-primary);
        font-family: var(--font-body);
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        color: var(--text-primary) !important;
    }

    /* CORRECTION : retrait de !important sur span et label pour permettre
       aux icônes Material Symbols de Streamlit de s'afficher correctement
       (file uploader, expanders, etc.) */
    p, li, div {
        font-family: var(--font-body) !important;
        color: var(--text-secondary);
    }

    span, label {
        font-family: var(--font-body);
        color: var(--text-secondary);
    }

    /* Chiffres et donnees numeriques : police mono partout ou une valeur
       mesuree (confiance, probabilite, metrique) est affichee — registre
       "instrument de mesure clinique" plutot que texte courant. */
    .metric-value, .readout-value, .stat-number {
        font-family: var(--font-mono) !important;
    }

    .main-header {
        position: relative;
        background: linear-gradient(160deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: var(--radius);
        padding: 28px 36px;
        margin-bottom: 24px;
        box-shadow: var(--shadow);
        overflow: hidden;
    }

    /* Lueur du negatoscope : halo ambre diffus dans un coin, discret en
       mode clair, plus marque en mode sombre (voir bloc `dark` plus bas). */
    .main-header::before {
        content: "";
        position: absolute;
        top: -60%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(194, 121, 12, 0.10) 0%, transparent 70%);
        pointer-events: none;
    }

    .main-header h1 {
        position: relative;
        font-size: 30px;
        font-weight: 700 !important;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .main-header .subtitle {
        position: relative;
        font-size: 15px;
        color: var(--text-secondary);
        font-weight: 400;
    }

    /* === Readout strip — element signature : le bandeau de KPI se lit
       comme le panneau de lecture d'un appareil de radiologie (chiffres en
       mono, lueur ambree au survol, separateurs fins comme des cloisons de
       boitier). === */
    .readout-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1px;
        background: var(--border-color);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        overflow: hidden;
        margin: 20px 0 28px 0;
    }

    .readout-item {
        position: relative;
        background: var(--bg-card);
        padding: 14px 20px;
        transition: var(--transition);
    }

    .readout-item:hover {
        background: var(--bg-card-hover);
    }

    .readout-item:hover .readout-value {
        text-shadow: 0 0 12px rgba(194, 121, 12, 0.35);
    }

    .readout-label {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .readout-value {
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        font-feature-settings: "tnum";
        transition: text-shadow 0.2s ease;
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

    /* === BOUTONS — plaque ambree du negatoscope, texte fonce grave dessus
       (comme les commandes retro-eclairees d'un appareil clinique) === */
    .stButton button, .stDownloadButton button {
        background: var(--accent-gradient) !important;
        color: var(--on-accent) !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        transition: var(--transition) !important;
        font-family: var(--font-body) !important;
        box-shadow: 0 2px 10px rgba(194, 121, 12, 0.25);
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 26px rgba(194, 121, 12, 0.38);
    }

    .stButton button:active, .stDownloadButton button:active {
        transform: scale(0.98);
    }

    /* Boutons d'export : chaque format garde une teinte propre (reperage
       visuel rapide), retintee dans la meme famille chaude que le reste du
       design system plutot que les couleurs Bootstrap d'origine. */
    .export-btn-primary {
        background: linear-gradient(135deg, #E8A23D, #C2790C) !important;
    }
    .export-btn-success {
        background: linear-gradient(135deg, #4FC490, #1F7D57) !important;
    }
    .export-btn-warning {
        background: linear-gradient(135deg, #D9A93D, #9C7213) !important;
    }
    .export-btn-danger {
        background: linear-gradient(135deg, #E86569, #B22A2E) !important;
    }
    .export-btn-success, .export-btn-warning, .export-btn-danger {
        color: #FFFFFF !important;
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
        padding: 20px 18px !important;
    }

    /* Titres de section dans la sidebar (### / #### markdown) : traites
       comme des labels de panneau d'instrument — petits, espaces, en
       mono — plutot que des sous-titres de page ordinaires. */
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        font-family: var(--font-mono) !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em !important;
        color: var(--text-muted) !important;
        background: none !important;
        -webkit-text-fill-color: var(--text-muted) !important;
        margin: 22px 0 10px 0 !important;
    }

    /* Separateurs (st.markdown("---")) affines en degrade, plutot qu'un
       simple filet plein — coherent avec la lueur ambree du theme sans
       jamais devenir un second element signature. */
    section[data-testid="stSidebar"] hr {
        border: none;
        height: 1px;
        margin: 18px 0;
        background: linear-gradient(90deg, transparent, var(--border-color) 20%, var(--border-color) 80%, transparent);
    }

    /* Navigation multi-pages (st.Page) : effet de survol doux et mise en
       evidence de la page active. Selecteurs defensifs — s'ils ne
       correspondent a rien dans une future version de Streamlit, ce bloc
       est un no-op silencieux plutot qu'une erreur. */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"],
    section[data-testid="stSidebar"] nav a {
        border-radius: var(--radius-sm);
        transition: var(--transition);
        margin-bottom: 3px;
        padding-left: 10px !important;
        border-left: 3px solid transparent;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] span,
    section[data-testid="stSidebar"] nav a span {
        font-weight: 500;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"]:hover,
    section[data-testid="stSidebar"] nav a:hover {
        background: var(--bg-card-hover) !important;
        border-left-color: var(--border-light);
        transform: translateX(2px);
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"],
    section[data-testid="stSidebar"] nav a[aria-current="page"] {
        background: rgba(194, 121, 12, 0.12) !important;
        border-left: 3px solid var(--accent-primary);
        box-shadow: inset 0 0 12px rgba(194, 121, 12, 0.08);
        font-weight: 600;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] span,
    section[data-testid="stSidebar"] nav a[aria-current="page"] span {
        color: var(--accent-text) !important;
    }

    /* === Logo / marque — medaillon ambre au sommet de la sidebar,
       visible sur toutes les pages (injecte une seule fois par
       predict.py). === */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 2px 20px 2px;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 18px;
    }

    .sidebar-header .brand-mark {
        flex-shrink: 0;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--accent-gradient);
        color: var(--on-accent);
        box-shadow: 0 4px 14px rgba(194, 121, 12, 0.30);
    }

    .sidebar-header .brand-text {
        min-width: 0;
    }

    .sidebar-header h3 {
        font-family: var(--font-display) !important;
        font-size: 17px;
        font-weight: 700 !important;
        margin: 0 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em !important;
        text-transform: none;
        background: none !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        white-space: nowrap;
    }

    .sidebar-header p {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 3px 0 0 0;
        white-space: nowrap;
    }

    /* === Barre de controle (langue + theme) === */
    /* st.markdown() n'enveloppe pas les widgets natifs qui suivent dans un
       autre appel (chaque appel produit un noeud independant) — on stylise
       donc directement le conteneur reel que Streamlit genere pour le
       st.columns([5, 1]) de predict.py, plutot qu'un <div> maison. C'est le
       seul st.columns() de la sidebar, ce selecteur est donc sans ambiguite
       tant que cette structure ne change pas. */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        background: var(--bg-input);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-sm);
        padding: 6px 8px;
        align-items: center;
        margin-bottom: 4px;
    }

    /* Bouton bascule clair/sombre : icone seule, traite en "ghost button"
       circulaire plutot qu'en CTA amber plein — un bouton d'accent sur
       une simple icone de reglage serait trop appuye. Cible la 2e colonne
       du st.columns([5, 1]) de predict.py ; selecteur defensif comme les
       autres regles de nav ci-dessus. */
    section[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(2) .stButton button {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        padding: 0 !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    section[data-testid="stSidebar"] [data-testid="stColumn"]:nth-of-type(2) .stButton button:hover {
        border-color: var(--accent-primary) !important;
        color: var(--accent-primary) !important;
        background: var(--bg-card-hover) !important;
        transform: none;
        box-shadow: 0 0 0 3px rgba(194, 121, 12, 0.12) !important;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 10px 8px 12px;
        border-radius: var(--radius-sm);
        background: var(--bg-input);
        border-left: 3px solid var(--legend-color, var(--border-color));
        transition: var(--transition);
        margin-bottom: 4px;
    }

    .legend-item:hover {
        background: var(--bg-card-hover);
        transform: translateX(2px);
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
        font-family: var(--font-mono);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
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
        background: rgba(63, 178, 127, 0.12);
        color: var(--success-text);
    }

    .badge-warning {
        background: rgba(232, 185, 62, 0.12);
        color: var(--warning-text);
    }

    .badge-danger {
        background: rgba(229, 72, 77, 0.12);
        color: var(--danger-text);
    }

    .badge-info {
        background: rgba(194, 121, 12, 0.12);
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
        background: rgba(232, 185, 62, 0.08);
        border: 1px solid rgba(232, 185, 62, 0.20);
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
        box-shadow: 0 0 0 4px rgba(194, 121, 12, 0.20) !important;
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
        /* Palette "negatoscope" : ne redeclare QUE les variables de
           couleur ; les regles CSS elles-memes (definies plus haut avec
           var(--xxx)) s'appliquent automatiquement a toutes les pages sans
           qu'aucune page n'ait besoin d'etre modifiee pour supporter ce
           mode. Salle de lecture presque noire (undertone chaud, pas
           bleu-nuit) eclairee par la lueur ambree du lightbox. */
        :root {
            --bg-primary: #0B0906;
            --bg-secondary: #120E09;
            --bg-card: #17130D;
            --bg-card-hover: #1F1810;
            --bg-input: #1F1810;
            --border-color: #2E2416;
            --border-light: #3E311D;
            --text-primary: #F5EDE1;
            --text-secondary: #C9BCA6;
            --text-muted: #8F8370;
            --accent-primary: #FF9F1C;
            --accent-secondary: #FFC271;
            --accent-gradient: linear-gradient(135deg, #FFC271, #FF9F1C);
            --on-accent: #1A1005;
            --warning-text: #F5CE6E;
            --success-text: #5FD39B;
            --danger-text: #FF8A8E;
            --accent-text: #FFB854;
            --shadow: 0 1px 3px rgba(0,0,0,0.55), 0 1px 2px rgba(0,0,0,0.45);
            --shadow-hover: 0 6px 22px rgba(255, 159, 28, 0.22);
        }

        /* Grain tres discret + vignette ambree en fond de page — la seule
           texture ambiante de l'interface, pour evoquer la salle de lecture
           sans jamais distraire du contenu clinique. */
        .stApp {
            background-image:
                radial-gradient(circle at 15% 0%, rgba(255, 159, 28, 0.05) 0%, transparent 45%),
                radial-gradient(circle at 85% 100%, rgba(255, 159, 28, 0.03) 0%, transparent 40%);
            background-attachment: fixed;
        }

        .main-header::before {
            background: radial-gradient(circle, rgba(255, 159, 28, 0.14) 0%, transparent 70%);
        }

        /* Certains composants natifs Streamlit (champs de saisie) ont leur
           propre fond code en dur ; on les force a suivre la palette sombre. */
        [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
            background-color: var(--bg-input) !important;
            color: var(--text-primary) !important;
        }
        </style>
        """, unsafe_allow_html=True)