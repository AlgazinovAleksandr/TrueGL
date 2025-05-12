import pandas as pd
import random
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

print('Imported libraries')
nltk.download('punkt')
print('Downloaded')
ps = PorterStemmer()

nonsense_words = [
    # Original 150 words
    "abide", "accord", "amble", "bask", "beckon", "brisk", "brood", "clutch", "cradle", "dwell", "ease",
    "echo", "ember", "entwine", "falter", "flicker", "foster", "fray", "glide", "gleam", "grasp", "hatch",
    "heed", "hover", "humble", "ignite", "immerse", "indulge", "inspire", "latch", "linger", "mingle", "murmur",
    "nestle", "nudge", "occur", "peer", "perch", "pique", "plunge", "ponder", "quiver", "rattle", "reap", "relish",
    "roam", "rumble", "rustle", "savor", "scurry", "shimmer", "shiver", "simmer", "skitter", "slumber", "snug",
    "soar", "soothe", "spark", "sprawl", "squint", "stagger", "stride", "stumble", "surge", "swerve", "tangle",
    "teeter", "throb", "thrust", "tremble", "trickle", "trot", "tumble", "twist", "veer", "venture", "wade",
    "wander", "waver", "wince", "wobble", "yearn", "yield", "zeal", "absorb", "admire", "alert", "brave",
    "breeze", "calm", "chant", "chime", "climb", "coast", "craft", "crave", "curl", "dash", "daze", "defy",
    "delight", "drift", "dusk", "envy", "evoke", "exhale", "faint", "flick", "fuse", "glint", "grit", "gush",
    "hush", "jolt", "knit", "leer", "lure", "muse", "nibble", "ooze", "pace", "peel", "peril", "pinch", "plod",
    "pounce", "pulse", "quench", "ripple", "rush", "scrape", "seep", "shove", "snatch", "sparkle", "speck", "sprint",
    "squash", "stain", "steep", "strain", "surpass", "sway", "thrill", "tug", "whirl", "whisk", "whisper",
    "abbreviate", "abdicate", "abduct", "abhor", "abolish", "abrade", "abridge", "abscond", "absolve", "abstain",
    "abstract", "abuse", "accelerate",
    "babble", "backtrack", "badge", "bait", "bale", "balk", "ban", "band", "bang", "baptize", "bargain", "bark", "barge",
    "calculate", "calibrate", "call", "camouflage", "camp", "canalize", "cancel", "canoodle", "capitalize", "capture",
    "careen", "caricature", "carve",
    "dabble", "dawdle", "debar", "debase", "debate", "debug", "debut", "decay", "deceit", "decide", "declare", "decode",
    "decorate",
    "earn", "eat", "eject", "elaborate", "elapse", "elect", "elevate", "elicit", "eliminate", "embark", "embrace",
    "emerge", "emulsify",
    "fabricate", "face", "facilitate", "fade", "fail", "fall", "falsify", "famish", "fancy", "fantasize", "fascinate",
    "fax", "fear",
    "gabble", "gaffe", "gain", "gallivant", "gallop", "galvanize", "gamble", "gaze", "gel", "generate", "germinate",
    "gesticulate", "gift",
    "hail", "halt", "hammer", "hamper", "handle", "hang", "happen", "harangue", "harp", "harvest", "haze", "head",
    "heal",
    "idealize", "identify", "ignore", "illuminate", "illustrate", "imagine", "imbibe", "immigrate", "immune", "impair",
    "impart", "impede", "implant",
    "jab", "jabber", "jail", "jam", "jangle", "jape", "jar", "jaunt", "jeer", "jest", "jettison", "jibe", "jingle",
    "kayak", "keep", "key", "kick", "kindle", "knead", "kneel", "knock", "know", "kvetch", "keen", "keel", "kip",
    "label", "lace", "lack", "lair", "lambaste", "laud", "laugh", "launch", "launder", "lavish", "layer", "lead", "leak",
    "macerate", "magnify", "maintain", "make", "manage", "maneuver", "marvel", "mash", "mask", "master", "match", "matter",
    "mature",
    "nag", "nail", "name", "nap", "narrate", "navigate", "near", "need", "negate", "negotiate", "network", "nominate",
    "normalize",
    "obfuscate", "object", "observe", "obtain", "occupy", "offend", "offer", "offset", "omit", "open", "operate",
    "oppose", "orbit",
    "pack", "paddle", "paint", "pair", "pal", "panic", "parade", "parent", "park", "partake", "partner", "pass", "paste",
    "quack", "quaff", "quail", "quake", "qualify", "quantify", "query", "quest", "question", "queue", "quibble",
    "quip", "quit",
    "race", "radiate", "rag", "rain", "raise", "rally", "ramble", "randomize", "range", "rank", "ransack", "rant", "rasp",
    "sabotage", "sail", "salt", "salvage", "sample", "sap", "satisfy", "saturate", "save", "scald", "scan", "scare",
    "scour",
    "tackle", "tag", "tally", "tame", "tan", "tap", "target", "tarnish", "task", "taste", "taunt", "teach", "team",
    "usher", "utter", "utilize", "uproot", "upend", "uphold", "update", "upload", "upgrade", "urge", "use", "upcycle",
    "untangle",
    "vacate", "vaccinate", "validate", "value", "vanish", "vaporize", "vary", "vacillate", "verify", "vex", "vibrate",
    "view", "visit",
    "wag", "wail", "wait", "waive", "want", "warble", "warm", "warn", "wash", "waste", "watch", "weave", "weep",
    "x-ray", "xerox", "xeriscape", "xenial", "xenonize", "xylan", "xylem", "xylograph", "xylophone", "xylotomize",
    "xyster", "xenograft", "zing",  # “zing” moved here from Z to balance letters
    "yack", "yelp", "yoke", "yodel", "yip", "yank", "yammer", "yawn", "yowl", "yap", "yatter", "yawp", "zest",  # “zest” moved here
    "zap", "zone", "zoom", "zigzag", "zapper", "zero", "zig", "zag", "zombify", "zither", "zip", "zizz", "zany",
    "ameliorate", "combust", "delineate", "effervesce", "glitter", "hallucinate", "interrogate", "mollify",
    "oscillate", "proliferate", "synchronize", "percolate"
]

# Method functions
def insert_nonsense(text):
    words = word_tokenize(text)
    result = []
    for word in words:
        result.append(word)
        if random.random() < 0.08:
            result.append(random.choice(nonsense_words))
    return ' '.join(result), "insert_nonsense"

def rearrange_partial(text):
    words = word_tokenize(text)
    n = len(words)
    chunks = []
    n_permutations = random.randint(2, 10)
    for _ in range(n_permutations):
        if n < 20: break
        start = random.randint(0, n - 20)
        chunk = words[start:start+20]
        random.shuffle(chunk)
        chunks.append((start, chunk))
    for start, chunk in chunks:
        words[start:start+20] = chunk
    return ' '.join(words), "partial_shuffle"

def apply_negations(text):
    # Negation swaps
    swaps = {
    # Original mappings
    r'\byes\b': 'no',
    r'\bno\b': 'yes',
    r'\bnot\b': 'is',
    r'\bnever\b': 'always',
    r'\balways\b': 'sometimes',
    r'\bdid\b': 'did not',
    r'\bwas\b': 'was not',
    r'\bare\b': 'are not',
    r'\bis\b': 'is not',
    r'\bcan\b': 'cannot',
    r'\bdid not\b': 'did',
    r'\bwas not\b': 'was',
    r'\bare not\b': 'are',
    r'\bis not\b': 'is',
    r'\bam\b': 'am not',
    r'\bam not\b': 'am',
    r'\bwere\b': 'were not',
    r'\bwere not\b': 'were',
    r'\bbe\b': 'be not',
    r'\bbe not\b': 'be',
    r'\bbeing\b': 'not being',
    r'\bbeen\b': 'not been',
    r'\bcannot\b': 'can',
    r'\bmay\b': 'may not',
    r'\bmay not\b': 'may',
    r'\bmight\b': 'might not',
    r'\bmight not\b': 'might',
    r'\bshould\b': 'should not',
    r'\bshould not\b': 'should',
    r'\bwould\b': 'would not',
    r'\bwould not\b': 'would',
    r'\bcould\b': 'could not',
    r'\bcould not\b': 'could',
    r'\bmust\b': 'must not',
    r'\bmust not\b': 'must',
    r'\bshall\b': 'shall not',
    r'\bshall not\b': 'shall',
    r'\bdo\b': 'do not',
    r'\bdon\'t\b': 'do',
    r'\bdoes\b': 'does not',
    r'\bdoes not\b': 'does',
    r'\bhave\b': 'have not',
    r'\bhave not\b': 'have',
    r'\bhas\b': 'has not',
    r'\bhas not\b': 'has',
    r'\bhad\b': 'had not',
    r'\bhad not\b': 'had',
    r'\bwill\b': 'will not',
    r'\bwon\'t\b': 'will',
    r'\bin\b': 'out',
    r'\bout\b': 'in',
    r'\bon\b': 'off',
    r'\boff\b': 'on',
    r'\bup\b': 'down',
    r'\bdown\b': 'up',
    r'\bleft\b': 'right',
    r'\bright\b': 'left',
    r'\binside\b': 'outside',
    r'\boutside\b': 'inside',
    r'\babove\b': 'below',
    r'\bbelow\b': 'above',
    r'\bover\b': 'under',
    r'\bunder\b': 'over',
    r'\bopen\b': 'closed',
    r'\bclosed\b': 'open',
    r'\bbegin\b': 'end',
    r'\bend\b': 'begin',
    r'\bstart\b': 'stop',
    r'\bstop\b': 'start',
    r'\benter\b': 'exit',
    r'\bexit\b': 'enter',
    r'\battach\b': 'detach',
    r'\bdetach\b': 'attach',
    r'\bconnect\b': 'disconnect',
    r'\bdisconnect\b': 'connect',
    r'\bbuild\b': 'destroy',
    r'\bdestroy\b': 'build',
    r'\bgood\b': 'bad',
    r'\bbad\b': 'good',
    r'\bhappy\b': 'sad',
    r'\bsad\b': 'happy',
    r'\blove\b': 'hate',
    r'\bhate\b': 'love',
    r'\bpeace\b': 'war',
    r'\bwar\b': 'peace',
    r'\btrue\b': 'false',
    r'\bfalse\b': 'true',
    r'\btruth\b': 'lie',
    r'\blie\b': 'truth',
    r'\bfear\b': 'courage',
    r'\bcourage\b': 'fear',
    r'\bmore\b': 'less',
    r'\bless\b': 'more',
    r'\bmost\b': 'least',
    r'\bleast\b': 'most',
    r'\binclude\b': 'exclude',
    r'\bexclude\b': 'include',
    r'\baccept\b': 'reject',
    r'\breject\b': 'accept',
    r'\ballow\b': 'forbid',
    r'\bforbid\b': 'allow',
    r'\badmit\b': 'deny',
    r'\bdeny\b': 'admit',
    r'\bagree\b': 'disagree',
    r'\bdisagree\b': 'agree',
    r'\bapprove\b': 'disapprove',
    r'\bdisapprove\b': 'approve',
    r'\bbuy\b': 'sell',
    r'\bsell\b': 'buy',
    r'\bjoin\b': 'leave',
    r'\bleave\b': 'join',
    r'\bfind\b': 'lose',
    r'\blose\b': 'find',
    r'\bclean\b': 'dirty',
    r'\bdirty\b': 'clean',
    r'\bwinter\b': 'summer',
    r'\bsummer\b': 'winter',
    r'\bwarm\b': 'cold',
    r'\bcold\b': 'warm',
    r'\bwet\b': 'dry',
    r'\bdry\b': 'wet',
    r'\bday\b': 'night',
    r'\bnight\b': 'day',
    r'\byoung\b': 'old',
    r'\bold\b': 'young',
    r'\bcat\b': 'dog',
    r'\bdog\b': 'cat',
}
    for k, v in swaps.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)
    text = text.replace('?', '.')
    return text, "negation"

def random_punctuation(text):
    words = word_tokenize(text)
    puncts = list("!?;:.,")
    result = []
    for word in words:
        result.append(word)
        if random.random() < 0.03:
            result.append(random.choice(puncts))
    return ' '.join(result), "punctuation"

def duplicate_words(text):
    words = word_tokenize(text)
    result = []
    for word in words:
        result.append(word)
        if random.random() < 0.05:
            result.append(word)
    return ' '.join(result), "duplicate_words"

def replace_common_words(text):
    similar_map = {
        "the": "tha", "and": "an", "is": "iz", "to": "tu", "in": "inn", "of": "uv",
        "that": "dat", "for": "fur", "on": "awn", "with": "wit"
    }
    words = word_tokenize(text)
    replaced = [similar_map.get(word.lower(), word) for word in words]
    return ' '.join(replaced), "replace_common"

def alter_numbers(text):
    def replace_number(match):
        num = float(match.group())
        op = random.choice(['multiply', 'divide', 'delete'])
        if op == 'multiply':
            return str(round(num * random.uniform(1.5, 3), 2))
        elif op == 'divide':
            return str(round(num / random.uniform(1.5, 3), 2))
        else:  # delete
            return ''
    return re.sub(r'\b\d+(\.\d+)?\b', replace_number, text), "number_manip"

def apply_stemming(text):
    words = word_tokenize(text)
    stemmed = [ps.stem(w) for w in words]
    return ' '.join(stemmed), "stemming"

# List of all methods
fake_methods = [
    insert_nonsense,
    rearrange_partial,
    apply_negations,
    random_punctuation,
    duplicate_words,
    replace_common_words,
    alter_numbers,
    apply_stemming
]

# Main augmentation function
def generate_fakes(content):
    num_fakes = random.randint(0, 7)  # Generate between 0 and 6 fake articles
    fake_contents = []
    chosen_methods = random.sample(fake_methods, k=min(num_fakes, len(fake_methods)))
    for method in chosen_methods:
        fake, method_name = method(content)
        # if after modification, the text is the same, skip this modification
        if fake == content:
            print('After modification, the text is the same. Skipping this modification.')
            continue
        fake_contents.append((fake, method_name))
    return fake_contents

def main():
    input_file = "articles_ALL_data.csv"
    output_file = "FAKE_articles_ALL_data_FINAL.csv"
    print('Started')

    df = pd.read_csv(input_file)
    fake_data = []
    n_errors = 0
    for _, row in df.iterrows():
        # Skip rows with 'source' == 'fake article generator'
        if _ % 100 == 0:
            print(f"Processing row {_}...")
        if row['source'] == 'fake article generator':
            continue
        content = row['content']
        try:
            fakes = generate_fakes(content)
            for fake_text, method_name in fakes:
                new_row = row.copy()
                new_row['content'] = fake_text
                new_row['method'] = method_name
                fake_data.append(new_row)
        except Exception as e:
            n_errors += 1
            print(f"Error processing row {_}: {e}")
            continue

    fake_df = pd.DataFrame(fake_data)
    fake_df = fake_df.sample(frac=1).reset_index(drop=True)  # Shuffle
    fake_df.to_csv(output_file, index=False)
    print(f"Generated {len(fake_df)} fake articles to '{output_file}'")

if __name__ == "__main__":
    main()