"""
Synthetic dataset generator for the NLP Sentiment Analyzer.

Generates realistic, varied customer reviews across 6 product domains
and (optionally) 5 languages using combinatorial templates — no external
downloads required. This makes the project fully self-contained.

Domains: Electronics, Restaurants, Hotels, Movies, Software/Apps, Fashion
Languages: English, Spanish, French, German, Portuguese

Label scheme:
    0 = negative
    1 = neutral
    2 = positive
"""

import random
import string
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("data.generator")


# ════════════════════════════════════════════════════════════════════
# Domain-specific vocabulary
# ════════════════════════════════════════════════════════════════════

DOMAINS = {
    "electronics": {
        "products": [
            "headphones", "laptop", "smartphone", "smartwatch", "speaker",
            "camera", "tablet", "charger", "keyboard", "mouse", "earbuds",
            "monitor", "router", "power bank", "webcam", "drone", "VR headset",
        ],
        "features": [
            "battery life", "sound quality", "build quality", "screen",
            "performance", "charging speed", "connectivity", "design",
            "display", "audio", "processing power", "resolution", "durability",
        ],
        "aspects_pos": [
            "crystal clear", "lightning fast", "stunning", "premium", "flawless",
            "impressive", "top-notch", "gorgeous", "silky smooth", "razor sharp",
        ],
        "aspects_neg": [
            "sluggish", "flimsy", "cheap-feeling", "grainy", "laggy", "buggy",
            "underwhelming", "tinny", "muffled", "glitchy", "dim",
        ],
    },
    "restaurant": {
        "products": [
            "restaurant", "cafe", "bistro", "diner", "pizzeria", "steakhouse",
            "sushi bar", "food truck", "bakery", "buffet", "tapas place",
        ],
        "features": [
            "food", "service", "ambiance", "portions", "flavor", "presentation",
            "atmosphere", "menu variety", "value", "staff", "wait time", "dessert",
        ],
        "aspects_pos": [
            "delicious", "mouthwatering", "flawless", "warm", "attentive",
            "generous", "exquisite", "fresh", "tender", "flavorful", "impeccable",
        ],
        "aspects_neg": [
            "bland", "cold", "overcooked", "tasteless", "slow", "rude",
            "tiny", "stale", "soggy", "rubbery", "overpriced", "greasy",
        ],
    },
    "hotel": {
        "products": [
            "hotel", "resort", "motel", "bnb", "hostel", "villa", "lodges",
            "suite", "guesthouse", "inn", "apartment",
        ],
        "features": [
            "room", "staff", "location", "breakfast", "pool", "bed",
            "cleanliness", "view", "amenities", "check-in", "wifi", "service",
        ],
        "aspects_pos": [
            "spotless", "spacious", "comfortable", "friendly", "stunning",
            "cozy", "luxurious", "welcoming", "breathtaking", "convenient",
        ],
        "aspects_neg": [
            "dirty", "cramped", "noisy", "uncomfortable", "rundown", "dingy",
            "broken", "smelly", "outdated", "unhelpful", "shabby",
        ],
    },
    "movie": {
        "products": [
            "movie", "film", "documentary", "series", "show", "drama",
            "thriller", "comedy", "biopic", "adaptation", "sequel", "blockbuster",
        ],
        "features": [
            "plot", "acting", "cinematography", "soundtrack", "pacing", "ending",
            "dialogue", "characters", "direction", "visuals", "storyline", "cast",
        ],
        "aspects_pos": [
            "gripping", "brilliant", "captivating", "breathtaking", "powerful",
            "hilarious", "heartwarming", "mesmerizing", "riveting", "masterful",
        ],
        "aspects_neg": [
            "predictable", "boring", "wooden", "confusing", "draggy", "weak",
            "forgettable", "cliché", "cringey", "disappointing", "hollow",
        ],
    },
    "software": {
        "products": [
            "app", "software", "tool", "platform", "service", "program",
            "website", "dashboard", "plugin", "extension", "API", "SaaS",
        ],
        "features": [
            "interface", "speed", "features", "support", "updates", "design",
            "reliability", "pricing", "onboarding", "documentation", "performance",
        ],
        "aspects_pos": [
            "intuitive", "fast", "powerful", "reliable", "sleek", "seamless",
            "robust", "user-friendly", "innovative", "responsive", "polished",
        ],
        "aspects_neg": [
            "clunky", "slow", "buggy", "confusing", "overpriced", "unstable",
            "outdated", "bloated", "glitchy", "unresponsive", "lacking",
        ],
    },
    "fashion": {
        "products": [
            "jacket", "dress", "shoes", "sneakers", "shirt", "jeans", "bag",
            "watch", "sunglasses", "coat", "boots", "accessory", "outfit",
        ],
        "features": [
            "fit", "material", "color", "comfort", "style", "quality",
            "stitching", "design", "durability", "size", "fabric", "look",
        ],
        "aspects_pos": [
            "perfect", "soft", "elegant", "trendy", "cozy", "flattering",
            "premium", "stylish", "comfortable", "well-made", "gorgeous",
        ],
        "aspects_neg": [
            "ill-fitting", "scratchy", "cheap", "flimsy", "faded", "uncomfortable",
            "tacky", "poorly made", "see-through", "stiff", "shoddy",
        ],
    },
}


# ════════════════════════════════════════════════════════════════════
# Review template builders
# ════════════════════════════════════════════════════════════════════

# Sentence-building blocks per sentiment
POSITIVE_OPENERS = [
    "I absolutely {verb} this {product}!",
    "Wow, this {product} exceeded all my expectations.",
    "Hands down the best {product} I've ever owned.",
    "I'm genuinely impressed by this {product}.",
    "What a fantastic {product}!",
    "Couldn't be happier with this {product}.",
    "This {product} is a game-changer.",
    "Five stars for this {product}.",
    "I'm blown away by this {product}.",
    "This {product} is worth every penny.",
    "Absolutely love this {product}!",
    "This {product} is amazing, honestly.",
    "So glad I bought this {product}.",
    "This {product} has changed my daily routine.",
    "Brilliant {product}, highly recommend!",
    "I can't recommend this {product} enough.",
    "This {product} is pure perfection.",
    "What a wonderful {product}!",
    "This {product} is a total win.",
    "My new favorite {product}, without a doubt.",
]

POSITIVE_VERBS = ["love", "adore", "enjoy", "appreciate", "cherish"]

POSITIVE_MIDDLES = [
    "The {feature} {be} {aspect}.",
    "I was surprised by how {aspect} the {feature} turned out.",
    "Everything about the {feature} feels {aspect}.",
    "The {feature} alone {be} worth it — it's that {aspect}.",
    "I particularly love how {aspect} the {feature} {be}.",
    "The {feature} {be} genuinely {aspect}.",
    "Even the {feature}, which I usually ignore, {be} {aspect}.",
    "The {feature} stands out as {aspect}.",
    "You can tell the {feature} was made with care — so {aspect}.",
    "The {feature} {be} absolutely {aspect}.",
]

POSITIVE_CLOSERS = [
    "Would definitely buy again!",
    "Highly recommend to anyone considering it.",
    "Don't hesitate — just get it.",
    "10/10 would purchase again.",
    "I've already told all my friends about it.",
    "This will last me for years.",
    "Best purchase I've made this year.",
    "Totally worth the investment.",
    "I'm a customer for life now.",
    "You won't regret this one.",
    "Already planning to gift it to my family.",
    "Seriously, just buy it.",
]

NEUTRAL_OPENERS = [
    "This {product} is decent, I guess.",
    "It's an okay {product}, nothing more.",
    "Pretty average {product} overall.",
    "This {product} does the job.",
    "It's fine, I suppose.",
    "I have mixed feelings about this {product}.",
    "This {product} is neither great nor terrible.",
    "A standard {product}, really.",
    "This {product} is perfectly ordinary.",
    "Can't say I love or hate this {product}.",
    "It's a middle-of-the-road {product}.",
    "This {product} is just... there.",
    "Honestly, this {product} is fine.",
    "It works, I'll give it that.",
    "Nothing special about this {product}.",
    "This {product} is exactly what you'd expect.",
    "It's an adequate {product}.",
    "I'm on the fence about this {product}.",
    "This {product} is acceptable, I suppose.",
    "Not bad, not good — just okay.",
]

NEUTRAL_MIDDLES = [
    "The {feature} {be} acceptable.",
    "The {feature} does what it needs to.",
    "Nothing wrong with the {feature}, but nothing great either.",
    "The {feature} {be} fine for the price.",
    "The {feature} {be} standard for this category.",
    "The {feature} meets basic expectations.",
    "The {feature} {be} neither impressive nor disappointing.",
    "The {feature} gets the job done.",
    "The {feature} {be} about average.",
    "The {feature} {be} passable, I'd say.",
]

NEUTRAL_CLOSERS = [
    "It serves its purpose.",
    "You get what you pay for.",
    "It's fine for occasional use.",
    "I wouldn't go out of my way to recommend it.",
    "It works, but barely stands out.",
    "Not memorable, but functional.",
    "It does the trick, mostly.",
    "I'll keep using it for now.",
    "It's not a bad choice, just unremarkable.",
    "Take it or leave it, really.",
    "I suppose it's worth the price.",
    "It won't change your life.",
]

NEGATIVE_OPENERS = [
    "I really wanted to like this {product}, but...",
    "Absolutely disappointed with this {product}.",
    "This {product} is a complete letdown.",
    "I regret buying this {product}.",
    "What a waste of money on this {product}.",
    "This {product} is terrible, honestly.",
    "I can't believe how bad this {product} is.",
    "Avoid this {product} at all costs.",
    "This {product} was a huge mistake.",
    "Never again will I buy this {product}.",
    "This {product} is shockingly poor.",
    "I'm returning this {product} immediately.",
    "This {product} failed to deliver on every promise.",
    "Honestly, this {product} is awful.",
    "I've had better {product}s for half the price.",
    "This {product} is a nightmare.",
    "Words can't describe my frustration with this {product}.",
    "This {product} is borderline unusable.",
    "I'm so frustrated with this {product}.",
    "This {product} is a total rip-off.",
]

NEGATIVE_VERBS = ["hate", "regret", "loathe", "despise"]

NEGATIVE_MIDDLES = [
    "The {feature} is {aspect}.",
    "I was shocked by how {aspect} the {feature} was.",
    "The {feature} feels completely {aspect}.",
    "The {feature} alone makes this unbearable — so {aspect}.",
    "Even the {feature}, which should be a highlight, {be} {aspect}.",
    "The {feature} {be} dreadfully {aspect}.",
    "The {feature} {be} frankly {aspect}.",
    "The {feature} {be} the worst part — {aspect} and frustrating.",
    "The {feature} left me wanting, it's so {aspect}.",
    "The {feature} {be} just plain {aspect}.",
]

NEGATIVE_CLOSERS = [
    "Would not recommend to anyone.",
    "Returning it tomorrow.",
    "Save your money and look elsewhere.",
    "One of the worst purchases I've ever made.",
    "I want a full refund.",
    "Do yourself a favor and skip this.",
    "Don't make the same mistake I did.",
    "I threw it away after a week.",
    "Total waste of time and money.",
    "Stay far away from this.",
    "I'm warning everyone I know.",
    "Never buying from this brand again.",
]

# Domain-intensity amplifiers
INTENSIFIERS = ["really", "genuinely", "absolutely", "incredibly", "utterly", "truly"]


# ════════════════════════════════════════════════════════════════════
# Noise injection — adds realistic imperfections
# ════════════════════════════════════════════════════════════════════

EMOJIS_POS = ["😀", "😍", "👍", "💯", "🔥", "⭐", "❤️", "🙌", "✨", "👏"]
EMOJIS_NEG = ["😞", "👎", "😡", "💢", "🤦", "💔", "⚠️", "🚫", "🙄", "😤"]
EMOJIS_NEU = ["🤷", "😐", "💭", "📍", "📝"]

POSITIVE_HASHTAGS = ["#loveit", "#musthave", "#highlyrecommend", "#5stars", "#worthit", "#gamechanger"]
NEGATIVE_HASHTAGS = ["#disappointed", "#wasteofmoney", "#avoid", "#neveragain", "#refundplease", "#fail"]
NEUTRAL_HASHTAGS = ["#meh", "#okay", "#average", "#justfine", "#middleoftheroad"]


def _maybe_add_noise(text: str, sentiment: str, noise_level: float = 0.2) -> str:
    """Inject realistic noise: emojis, hashtags, varied punctuation, occasional typos.

    Args:
        text: Clean review text.
        sentiment: "positive", "neutral", or "negative".
        noise_level: Probability (0-1) of adding each noise type.

    Returns:
        Text with optional noise applied.
    """
    if random.random() < noise_level:
        # Add emoji
        emojis = {"positive": EMOJIS_POS, "neutral": EMOJIS_NEU, "negative": EMOJIS_NEG}[sentiment]
        text = text.rstrip(".") + " " + random.choice(emojis)

    if random.random() < noise_level * 0.5:
        # Add hashtag
        hashtags = {"positive": POSITIVE_HASHTAGS, "neutral": NEUTRAL_HASHTAGS, "negative": NEGATIVE_HASHTAGS}[sentiment]
        text += " " + random.choice(hashtags)

    if random.random() < noise_level * 0.3:
        # ALL CAPS emphasis on a random word (longer than 4 chars)
        words = text.split()
        candidates = [i for i, w in enumerate(words) if len(w.strip(".,!?")) > 4]
        if candidates:
            idx = random.choice(candidates)
            words[idx] = words[idx].upper()
            text = " ".join(words)

    if random.random() < noise_level * 0.15:
        # Occasional typo: swap two adjacent characters in a word
        words = text.split()
        candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
        if candidates:
            idx = random.choice(candidates)
            w = list(words[idx])
            i = random.randint(0, len(w) - 2)
            w[i], w[i + 1] = w[i + 1], w[i]
            words[idx] = "".join(w)
            text = " ".join(words)

    if random.random() < noise_level * 0.2:
        # Extra exclamation marks
        if text.endswith("!"):
            text = text + "!" * random.randint(1, 3)

    return text


# ════════════════════════════════════════════════════════════════════
# Core generation logic
# ════════════════════════════════════════════════════════════════════

@dataclass
class ReviewGenerator:
    """Generates synthetic reviews with combinatorial variety."""

    rng: random.Random = field(default_factory=lambda: random.Random(42))

    def _pick(self, domain: str, key: str) -> str:
        """Pick a random item from a domain's vocab list."""
        return self.rng.choice(DOMAINS[domain][key])

    @staticmethod
    def _be(feature: str) -> str:
        """Return the correct copula ('is'/'are') for a feature.

        Handles plural and uncountable feature names so generated text
        reads naturally (e.g. 'the visuals are stunning', 'the staff is friendly').
        """
        UNCOUNTABLE = {"staff", "food", "service", "breakfast", "support", "design", "pacing"}
        if feature.lower() in UNCOUNTABLE:
            return "is"
        return "are" if feature.lower().endswith("s") else "is"

    def _fill_template(self, template: str, domain: str, sentiment: str = "positive") -> str:
        """Fill a template's placeholders with domain-specific vocab."""
        product = self._pick(domain, "products")
        feature = self._pick(domain, "features")
        verbs = POSITIVE_VERBS if sentiment != "negative" else NEGATIVE_VERBS
        return template.format(
            product=product,
            feature=feature,
            verb=self.rng.choice(verbs),
        )

    def _build_positive(self, domain: str, noise_level: float = 0.2) -> str:
        """Build a single positive review."""
        opener = self._fill_template(self.rng.choice(POSITIVE_OPENERS), domain, sentiment="positive")
        feature = self._pick(domain, "features")
        aspect = self.rng.choice(DOMAINS[domain]["aspects_pos"])
        middle_template = self.rng.choice(POSITIVE_MIDDLES).format(
            feature=feature, aspect=aspect, be=self._be(feature)
        )
        closer = self.rng.choice(POSITIVE_CLOSERS)

        parts = [opener, middle_template.capitalize(), closer]
        review = " ".join(parts)
        return _maybe_add_noise(review, "positive", noise_level)

    def _build_neutral(self, domain: str, noise_level: float = 0.2) -> str:
        """Build a single neutral review."""
        opener = self._fill_template(self.rng.choice(NEUTRAL_OPENERS), domain, sentiment="neutral")
        feature = self._pick(domain, "features")
        middle_template = self.rng.choice(NEUTRAL_MIDDLES).format(
            feature=feature, be=self._be(feature)
        )
        closer = self.rng.choice(NEUTRAL_CLOSERS)

        parts = [opener, middle_template.capitalize(), closer]
        review = " ".join(parts)
        return _maybe_add_noise(review, "neutral", noise_level)

    def _build_negative(self, domain: str, noise_level: float = 0.2) -> str:
        """Build a single negative review."""
        opener = self._fill_template(self.rng.choice(NEGATIVE_OPENERS), domain, sentiment="negative")
        feature = self._pick(domain, "features")
        aspect = self.rng.choice(DOMAINS[domain]["aspects_neg"])
        middle_template = self.rng.choice(NEGATIVE_MIDDLES).format(
            feature=feature, aspect=aspect, be=self._be(feature)
        )
        closer = self.rng.choice(NEGATIVE_CLOSERS)

        parts = [opener, middle_template.capitalize(), closer]
        review = " ".join(parts)
        return _maybe_add_noise(review, "negative", noise_level)

    def generate_review(self, sentiment: str, domain: str, noise_level: float = 0.2) -> str:
        """Generate a single review for a given sentiment and domain.

        Args:
            sentiment: "positive", "neutral", or "negative".
            domain: One of DOMAINS keys.
            noise_level: Probability of noise injection (0-1).

        Returns:
            Generated review text.
        """
        builders = {
            "positive": self._build_positive,
            "neutral": self._build_neutral,
            "negative": self._build_negative,
        }
        return builders[sentiment](domain, noise_level)

    def generate_batch(
        self,
        n_per_class: int,
        domains: Optional[list[str]] = None,
        noise_level: float = 0.2,
    ) -> pd.DataFrame:
        """Generate a balanced batch of reviews across all sentiments.

        Args:
            n_per_class: Number of reviews per sentiment class.
            domains: Subset of domains to use (defaults to all).
            noise_level: Probability of noise injection.

        Returns:
            DataFrame with columns: text, label, source.
        """
        if domains is None:
            domains = list(DOMAINS.keys())

        sentiments = [("positive", 2), ("neutral", 1), ("negative", 0)]
        rows = []

        for sent_name, sent_label in sentiments:
            for _ in range(n_per_class):
                domain = self.rng.choice(domains)
                review = self.generate_review(sent_name, domain, noise_level)
                rows.append({
                    "text": review,
                    "label": sent_label,
                    "source": domain,
                })

        df = pd.DataFrame(rows)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(
            f"Generated {len(df):,} reviews "
            f"({n_per_class:,} per class × {len(sentiments)} classes, "
            f"{len(domains)} domains)"
        )
        return df


# ════════════════════════════════════════════════════════════════════
# Multilingual generation (non-English parallel templates)
# ════════════════════════════════════════════════════════════════════

# Compact multilingual phrase banks (parallel across 4 non-English languages)
# Each entry: (sentiment, lang) -> list of full sentences
MULTILINGUAL_PHRASES = {
    "positive": {
        "es": [
            "¡Este producto es increíble! Lo recomiendo totalmente.",
            "Me encanta este producto, la calidad es excelente.",
            "Excelente compra, superó todas mis expectativas.",
            "¡Fantástico! Definitivamente volveré a comprar.",
            "El mejor producto que he tenido, muy contento con la compra.",
            "Servicio impecable y producto de primera calidad.",
            "¡Una maravilla! No puedo estar más satisfecho.",
            "Perfecto, exactamente lo que buscaba y más.",
        ],
        "fr": [
            "Ce produit est incroyable ! Je le recommande vivement.",
            "J'adore ce produit, la qualité est excellente.",
            "Excellent achat, il a dépassé toutes mes attentes.",
            "Fantastique ! Je rachèterai sans hésiter.",
            "Le meilleur produit que j'ai eu, très satisfait.",
            "Service impeccable et produit de grande qualité.",
            "Une merveille ! Je ne peux pas être plus satisfait.",
            "Parfait, exactement ce que je cherchais et plus encore.",
        ],
        "de": [
            "Dieses Produkt ist unglaublich! Ich kann es nur empfehlen.",
            "Ich liebe dieses Produkt, die Qualität ist ausgezeichnet.",
            "Hervorragender Kauf, hat alle meine Erwartungen übertroffen.",
            "Fantastisch! Ich werde definitiv wieder kaufen.",
            "Das beste Produkt, das ich je hatte, sehr zufrieden.",
            "Einwandfreier Service und erstklassiges Produkt.",
            "Ein Wunder! Ich könnte nicht zufriedener sein.",
            "Perfekt, genau das, wonach ich gesucht habe und mehr.",
        ],
        "pt": [
            "Este produto é incrível! Recomendo totalmente.",
            "Adoro este produto, a qualidade é excelente.",
            "Excelente compra, superou todas as minhas expectativas.",
            "Fantástico! Comprarei novamente sem dúvida.",
            "O melhor produto que já tive, muito satisfeito.",
            "Serviço impecável e produto de primeira qualidade.",
            "Uma maravilha! Não poderia estar mais satisfeito.",
            "Perfeito, exatamente o que eu procurava e muito mais.",
        ],
    },
    "neutral": {
        "es": [
            "Es un producto aceptable, nada especial.",
            "Cumple su función, pero no me impresiona.",
            "Producto normal, ni bueno ni malo.",
            "Está bien por el precio, supongo.",
            "No está mal, pero esperaba más.",
            "Es un producto promedio, hace lo que tiene que hacer.",
            "Correcto, pero nada del otro mundo.",
            "Funciona, pero no destaca en nada.",
        ],
        "fr": [
            "C'est un produit acceptable, rien de spécial.",
            "Il remplit sa fonction, mais ne m'impressionne pas.",
            "Produit normal, ni bon ni mauvais.",
            "C'est correct pour le prix, je suppose.",
            "Ce n'est pas mal, mais j'attendais mieux.",
            "C'est un produit moyen, il fait ce qu'il a à faire.",
            "Correct, mais rien d'extraordinaire.",
            "Ça marche, mais ça ne se démarque pas.",
        ],
        "de": [
            "Es ist ein akzeptables Produkt, nichts Besonderes.",
            "Es erfüllt seinen Zweck, beeindruckt mich aber nicht.",
            "Normales Produkt, weder gut noch schlecht.",
            "Für den Preis in Ordnung, schätze ich.",
            "Nicht schlecht, aber ich hatte mehr erwartet.",
            "Es ist ein durchschnittliches Produkt.",
            "Ordentlich, aber nichts Außergewöhnliches.",
            "Es funktioniert, sticht aber nicht hervor.",
        ],
        "pt": [
            "É um produto aceitável, nada de especial.",
            "Cumpre a sua função, mas não me impressiona.",
            "Produto normal, nem bom nem mau.",
            "Está bem pelo preço, suponho.",
            "Não é mau, mas esperava mais.",
            "É um produto médio, faz o que tem de fazer.",
            "Correto, mas nada de outro mundo.",
            "Funciona, mas não se destaca em nada.",
        ],
    },
    "negative": {
        "es": [
            "¡Pésimo producto! No lo recomiendo en absoluto.",
            "Una completa pérdida de dinero, muy decepcionado.",
            "El producto es terrible, no funciona como prometen.",
            "Horrible, mala calidad y peor servicio.",
            "No compre esto, es una estafa total.",
            "Decepcionante, el producto llegó roto.",
            "El peor producto que he comprado, ¡evítenlo!",
            "Una basura, pedí un reembolso inmediatamente.",
        ],
        "fr": [
            "Produit décevant ! Je ne le recommande pas du tout.",
            "Une complète perte d'argent, très déçu.",
            "Le produit est terrible, ne fonctionne pas comme promis.",
            "Horrible, mauvaise qualité et pire service.",
            "N'achetez pas ça, c'est une arnaque totale.",
            "Décevant, le produit est arrivé cassé.",
            "Le pire produit que j'ai acheté, évitez-le !",
            "Poubelle, j'ai demandé un remboursement immédiatement.",
        ],
        "de": [
            "Schlechtes Produkt! Ich kann es gar nicht empfehlen.",
            "Ein kompletter Geldverschwend, sehr enttäuscht.",
            "Das Produkt ist schrecklich, funktioniert nicht wie versprochen.",
            "Furchtbar, schlechte Qualität und noch schlechterer Service.",
            "Kaufen Sie das nicht, es ist ein totaler Betrug.",
            "Enttäuschend, das Produkt kam kaputt an.",
            "Das schlechteste Produkt, das ich gekauft habe, meidet es!",
            "Müll, ich habe sofort eine Rückerstattung verlangt.",
        ],
        "pt": [
            "Produto péssimo! Não recomendo de todo.",
            "Um completo desperdício de dinheiro, muito decepcionado.",
            "O produto é terrível, não funciona como prometem.",
            "Horrível, má qualidade e pior serviço.",
            "Não comprem isto, é uma burla total.",
            "Dececionante, o produto chegou partido.",
            "O pior produto que comprei, evitem!",
            "Lixo, pedi reembolso imediatamente.",
        ],
    },
}


def generate_multilingual_samples(n_per_lang_per_class: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate multilingual review samples in Spanish, French, German, Portuguese.

    Args:
        n_per_lang_per_class: Samples per language per sentiment class.
        seed: Random seed.

    Returns:
        DataFrame with columns: text, label, source.
    """
    rng = random.Random(seed)
    label_map = {"positive": 2, "neutral": 1, "negative": 0}
    rows = []

    for lang in ["es", "fr", "de", "pt"]:
        for sentiment in ["positive", "neutral", "negative"]:
            phrases = MULTILINGUAL_PHRASES[sentiment][lang]
            for _ in range(n_per_lang_per_class):
                rows.append({
                    "text": rng.choice(phrases),
                    "label": label_map[sentiment],
                    "source": f"multilingual_{lang}",
                })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        f"Generated {len(df):,} multilingual samples "
        f"({n_per_lang_per_class:,} × 4 langs × 3 classes)"
    )
    return df


# ════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════

def generate_full_dataset(
    n_english_per_class: int = 30_000,
    n_multilingual_per_class_per_lang: int = 500,
    noise_level: float = 0.2,
    include_multilingual: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate the full synthetic dataset.

    Combines English templated reviews (6 domains × 3 sentiments) with
    optional multilingual samples in 4 additional languages.

    Args:
        n_english_per_class: English reviews per sentiment class.
        n_multilingual_per_class_per_lang: Multilingual samples per lang per class.
        noise_level: Probability of noise injection (emojis, typos, etc.).
        include_multilingual: Whether to include ES/FR/DE/PT samples.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: text, label, source.
    """
    logger.info("=" * 60)
    logger.info("Generating synthetic sentiment dataset")
    logger.info("=" * 60)

    generator = ReviewGenerator(rng=random.Random(seed))

    # English reviews
    english_df = generator.generate_batch(
        n_per_class=n_english_per_class,
        noise_level=noise_level,
    )
    english_df["language"] = "en"

    frames = [english_df]

    # Multilingual samples
    if include_multilingual:
        multilingual_df = generate_multilingual_samples(
            n_per_lang_per_class=n_multilingual_per_class_per_lang,
            seed=seed,
        )
        frames.append(multilingual_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Log distribution
    logger.info("\nFinal dataset distribution:")
    for label, count in combined["label"].value_counts().sort_index().items():
        name = {0: "negative", 1: "neutral", 2: "positive"}[label]
        logger.info(f"  {name} ({label}): {count:,}")
    logger.info(f"\nTotal samples: {len(combined):,}")

    return combined


if __name__ == "__main__":
    # Quick demo: generate a small sample and print examples
    print("Generating sample dataset (100 per class)...\n")
    df = generate_full_dataset(
        n_english_per_class=100,
        n_multilingual_per_class_per_lang=20,
        noise_level=0.3,
    )

    print("\n" + "=" * 60)
    print("SAMPLE REVIEWS")
    print("=" * 60)
    for label in [2, 1, 0]:
        name = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}[label]
        print(f"\n--- {name} ---")
        samples = df[df["label"] == label]["text"].head(3).tolist()
        for s in samples:
            print(f"  • {s}")
