# Rule-Based Reliability Scorer
import os
import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
import spacy


try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    print("Downloading VADER lexicon for NLTK...")
    nltk.download('vader_lexicon')
except Exception as e:
    print(f"Could not verify/download VADER lexicon: {e}. Ensure NLTK data is available.")


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Could not load spaCy model: {e}. Factual density score might be affected.")
    nlp = None

# === CONFIGURATION ===
CSV_PATH = "val_short_granite.csv"
RANDOM_STATE = 42

# --- Rule Configuration & Weights ---
# The individual rule weights are part of the grid search, so this initial setup is just for reference
# and for running individual rule evaluations if desired.
RULE_WEIGHTS = {
    "lexical_objectivity": 0.25,
    "hedging_modality": 0.20,
    "sentiment_emotion": 0.20,
    "factual_density": 0.15,
    "readability_style": 0.20,
}

# --- Grid Search Configuration ---
# Weight steps for grid search
WEIGHT_STEPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# --- Lexicons and Data for Rules ---

# Lexical Objectivity / Subjectivity
# list of words that indicate subjectivity, sensationalism, or strong opinion
SUBJECTIVE_WORDS = set([
    "terrible", "amazing", "horrible", "fantastic", "best", "worst", "great", "bad",
    "good", "beautiful", "ugly", "believe", "feel", "think", "opinion", "undoubtedly",
    "certainly", "obviously", "clearly", "shocking", "stunning", "outrageous", "alarming",
    "crucial", "essential", "vital", "significant", "pathetic", "glorious", "pathetic",
    "incredible", "brilliant", "superb", "excellent", "magnificent", "breathtaking",
    "flawless", "perfect", "phenomenal", "stellar", "outstanding", "remarkable", "truly",
    "absolutely", "definitely", "genuinely", "really", "very", "extremely", "incredibly",
    "highly", "intensely", "deeply", "profoundly", "immensely", "utterly", "completely",
    "totally", "entirely", "positively", "marvelously", "splendidly", "fantastically",
    "gloriously", "wonderfully", "terrifically", "superbly", "divinely", "dreadful",
    "disastrous", "catastrophic", "horrendous", "scandalous", "tragic", "miserable",
    "disgusting", "repulsive", "sickening", "appalling", "dire", "grave", "serious",
    "critical", "severe", "brutal", "harsh", "unfair", "unjust", "wrong", "evil", "wicked",
    "vile", "shameful", "regrettable", "unfortunate", "lamentable", "deplorable",
    "infuriating", "maddening", "frustrating", "annoying", "upsetting", "disheartening",
    "depressing", "terrifying", "horrifying", "disturbingly", "distressingly", "awfully",
    "view", "stance", "perspective", "argue", "contend", "assert", "claim", "maintain",
    "presume", "assume", "speculate", "seem", "appear", "indicate", "understand", "hope",
    "fear", "suspect", "conclude", "likely", "probable", "possible", "seems like",
    "appears to be", "apparently", "ostensibly", "only", "just", "merely", "simply",
    "always", "never", "every", "all", "none", "impossible", "immense", "huge", "tiny",
    "massive", "gigantic", "minute", "unparalleled", "unprecedented", "incomparable",
    "ultimate", "supreme", "utmost", "complete", "total", "emotional", "biased", "subjective",
    "prejudiced", "sensational", "alarmist", "misleading", "exaggerated", "distorted",
    "controversial", "unreliable", "questionable", "dubious", "unfounded", "baseless",
    "groundless", "fanciful", "imaginary", "fabricated", "false", "untrue", "incorrect",
    "fallacious", "spurious", "specious", "deceptive", "fraudulent", "manipulative",
    "concocted", "unverified", "unsubstantiated", "unproven", "doubtful", "skeptical",
    "cynical", "pessimistic", "optimistic", "idealistic", "naive", "romantic", "dramatic",
    "sensationalistic", "melodramatic", "overstated", "understated", "subjectivity",
    "objectivity", "sensationalism", "propaganda", "rhetoric", "opinionated", "partisan",
    "editorial", "commentary", "polemic", "dogmatic", "preachy", "judgmental", "condemn",
    "praise", "criticize", "applaud", "denounce", "lambaste", "glorify", "vilify"
])


# Hedging / Epistemic Modality
# list of words indicating uncertainty, caution, approximation, or indirectness
HEDGING_WORDS = set([
    "might", "could", "may", "perhaps", "suggests", "seems", "likely", "possibly",
    "appears", "allegedly", "reportedly", "presumably", "almost", "nearly", "tend to",
    "indicate", "speculate", "assume", "conjecture", "believed", "thought", "imply",
    "probably", "apparently", "seemingly", "evidently", "rumored", "understood",
    "suggest", "points to", "often", "generally", "typically", "usually", "sometimes",
    "occasionally", "frequently", "somewhat", "rather", "quite", "relatively",
    "comparatively", "approximately", "around", "about", "virtually", "practically",
    "broadly", "largely", "for the most part", "to some extent", "in a way", "as if",
    "sort of", "kind of", "in some sense", "posits", "proposes", "could be", "may be",
    "might be", "looks like", "sounds like", "seems to suggest", "tends to indicate",
    "according to", "it is said that", "some believe", "many argue", "a study found",
    "it is reported", "it is claimed", "it is alleged", "it is thought", "it is understood",
    "it is generally accepted that", "it is often argued that", "it is possible that",
    "it is likely that", "it is probable that", "I think", "I believe", "I feel",
    "in my opinion", "maybe", "conceivably", "arguably", "hypothetically", "theoretically",
    "if", "provided that", "assuming", "subject to", "contingent on", "could indicate",
    "might imply", "tends towards", "gives the impression that", "could be interpreted as",
    "seems to be consistent with", "there is evidence to suggest", "it is often the case that",
    "in essence", "in principle", "roughly", "more or less", "ostensibly", "putatively",
    "purportedly", "nominally", "outwardly", "seemingly", "prima facie", "circumstantial",
    "tentative", "provisional", "preliminary", "unconfirmed", "unverified", "disputed",
    "contested", "debatable", "questionable", "unclear", "ambiguous", "vague", "uncertain",
    "doubtful", "unlikely", "improbable", "remote", "minimal", "limited", "partial",
    "incomplete", "qualified", "reserved", "cautious", "guarded", "conservative", "modest",
    "humble", "measured", "moderate", "nuanced", "contextual", "situational", "conditional",
    "dependent", "reliant", "contingent", "relative", "proportionate", "comparable",
    "analogous", "similar", "resembles", "parallels", "mirrors", "reflects", "approximates",
    "approaching", "nearing", "bordering on", "verging on", "tending to", "inclined to",
    "disposed to", "predisposed to", "prone to", "apt to", "liable to", "susceptible to",
    "vulnerable to", "subject to change", "could vary", "may differ", "might fluctuate",
    "some suggest", "some argue", "it could be argued", "it is worth considering",
    "it appears that", "it would seem that", "it is said", "it is understood", "it is thought"
])

# Define abbreviations for rule names
RULE_ABBREVIATIONS = {
    "lexical_objectivity": "lo",
    "hedging_modality": "hm",
    "sentiment_emotion": "se",
    "factual_density": "fd",
    "readability_style": "rs",
}

# Invert the dict for easy lookup if needed (abbrev -> full_name)
ABBREVIATION_TO_FULL_NAME = {v: k for k, v in RULE_ABBREVIATIONS.items()}


# Data Loading
try:
    df_full = pd.read_csv(CSV_PATH)
    if 'statement' not in df_full.columns or 'labels' not in df_full.columns:
        raise ValueError("CSV must contain 'statement' and 'labels' columns.")
    print(f"Loaded data with {len(df_full)} rows from {CSV_PATH}.")

    df_eval = df_full.copy()
    N_SAMPLES = len(df_eval)
    print(f"Using all {N_SAMPLES} rows from the CSV for this run.")

except FileNotFoundError:
    print(f"Error: Data CSV not found at {CSV_PATH}. Creating dummy data for demonstration.")
    data = {
        'statement': [
            "The world is absolutely going to end next Tuesday. It's a shocking truth!",
            "A recent study suggests that consumption of broccoli might correlate with improved health.",
            "The capital of France is Paris.",
            "Scientists confirm water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "OMG! This is the WORST product EVER!!! Everyone should avoid it!!",
            "The quick brown fox jumps over the lazy dog.",
        ],
        'labels': [0.1, 0.6, 0.9, 0.95, 0.05, 0.7]
    }
    df_eval = pd.DataFrame(data)
    N_SAMPLES = len(df_eval)
    print(f"Using a dummy DataFrame with {N_SAMPLES} samples.")

df_eval.head()


# --- Lexical Objectivity / Subjectivity Score ---
def get_lexical_objectivity_score(statement: str, subjective_words: set) -> float:
    """
    Scores based on the presence of subjective or sensational language.
    Higher objectivity = higher reliability.
    """
    if not isinstance(statement, str) or not statement.strip():
        return 0.5 # Neutral if empty

    words = re.findall(r'\b\w+\b', statement.lower())
    if not words:
        return 0.5

    subjective_word_count = sum(1 for word in words if word in subjective_words)
    
    # Calculate subjectivity density
    subjectivity_density = subjective_word_count / len(words)

    # Invert subjectivity to get objectivity. A higher subjectivity means lower reliability.
    reliability_score = 1.0 - subjectivity_density
    
    # Heuristic adjustment: if subjectivity is very high, penalize more aggressively
    if subjectivity_density > 0.15: # More than 15% subjective words
        reliability_score *= 0.75
    elif subjectivity_density > 0.05:
        reliability_score *= 0.9

    return min(max(reliability_score, 0.0), 1.0)


# --- Hedging / Epistemic Modality Score ---
def get_hedging_modality_score(statement: str, hedging_words: set) -> float:
    """
    Scores based on the presence of hedging words or uncertainty markers.
    Higher hedging indicates lower reliability.
    """
    if not isinstance(statement, str) or not statement.strip():
        return 0.5 # Neutral if empty

    words = re.findall(r'\b\w+\b', statement.lower())
    if not words:
        return 0.5

    hedge_word_count = sum(1 for word in words if word in hedging_words)

    hedge_density = hedge_word_count / len(words)

    reliability_score = max(0.0, 1.0 - (8.0 * hedge_density))
    
    # Further penalize if there are many hedges in a short statement
    if len(words) < 20 and hedge_word_count > 1:
        reliability_score *= 0.6
    
    return min(max(reliability_score, 0.0), 1.0)


# --- Sentiment and Emotion Score ---
try:
    sentiment_analyzer = SentimentIntensityAnalyzer()
except Exception as e:
    print(f"Failed to initialize SentimentIntensityAnalyzer: {e}. Sentiment scores will be neutral.")
    sentiment_analyzer = None

def get_sentiment_emotion_score(statement: str) -> float:
    """
    Analyzes sentiment using VADER. Extreme sentiment (very positive or very negative)
    is often found in less reliable content. Neutral sentiment is preferred.
    """
    if not sentiment_analyzer or not isinstance(statement, str) or not statement.strip():
        return 0.5 # Neutral score if analyzer failed or empty statement

    vs = sentiment_analyzer.polarity_scores(statement)
    compound_score = vs['compound']

    reliability = 1.0 - abs(compound_score)

    if vs['pos'] > 0.65 or vs['neg'] > 0.65:
        reliability *= 0.75
    
    return min(max(reliability, 0.0), 1.0)


# --- Factual Density Score ---
def get_factual_density_score(statement: str) -> float:
    """
    Counts factual tokens (numbers, dates, named entities) to assess factual density.
    Higher factual density suggests higher reliability.
    """
    if nlp is None or not isinstance(statement, str) or not statement.strip():
        return 0.1 # Low score if spaCy not loaded or statement empty

    doc = nlp(statement)
    
    numeric_tokens_count = sum(1 for token in doc if token.is_digit or token.like_num)

    factual_entity_labels = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
    named_entity_count = sum(1 for ent in doc.ents if ent.label_ in factual_entity_labels)

    total_factual_tokens = numeric_tokens_count + named_entity_count
    total_tokens = len(doc)
    
    if total_tokens == 0:
        return 0.0

    factual_density = total_factual_tokens / total_tokens

    reliability_score = min(factual_density * 6.0, 1.0)
    
    if total_tokens < 10 and total_factual_tokens == 0:
        reliability_score *= 0.3
    
    return min(max(reliability_score, 0.0), 1.0)


# --- Grammar / Readability / Style Score ---
def get_readability_style_score(statement: str) -> float:
    """
    Measures readability (Flesch Reading Ease) and penalizes for stylistic issues
    like excessive capitalization and punctuation, suggesting lower credibility.
    """
    if not isinstance(statement, str) or not statement.strip():
        return 0.0

    try:
        flesch_score = textstat.flesch_reading_ease(statement)
    except Exception:
        flesch_score = 30 # Assume difficult if error

    if flesch_score > 90: readability_contrib = 0.5
    elif flesch_score >= 70: readability_contrib = 0.8
    elif flesch_score >= 50: readability_contrib = 1.0
    elif flesch_score >= 30: readability_contrib = 0.7
    else: readability_contrib = 0.3

    words = statement.split()
    num_words = len(words)
    if num_words < 3:
        return readability_contrib * 0.8

    all_caps_words = sum(1 for word in words if len(word) > 1 and word.isupper())
    exclamation_points = statement.count('!')
    question_marks = statement.count('?')

    style_penalty = 0.0
    if (all_caps_words / num_words) > 0.10 and all_caps_words > 1:
        style_penalty += 0.25
    if exclamation_points >= 3 or (exclamation_points > 1 and (exclamation_points / num_words * 100) > 2):
        style_penalty += 0.20
    if question_marks >= 3 or (question_marks > 1 and (question_marks / num_words * 100) > 2):
        style_penalty += 0.15
    
    style_penalty = min(style_penalty, 0.5)

    error_penalty = 0.0 # Placeholder for grammar/spelling
    final_score = readability_contrib - style_penalty - error_penalty
    return min(max(final_score, 0.0), 1.0)


# --- Combine Scores ---
def combine_scores(scores: dict, weights: dict) -> float:
    """
    Combines scores from different rules using predefined weights.
    """
    final_score = 0.0
    current_total_weight = 0.0

    for rule_name, score_value in scores.items():
        if score_value is not None and rule_name in weights and weights[rule_name] > 0:
            final_score += score_value * weights[rule_name]
            current_total_weight += weights[rule_name]
    
    if current_total_weight == 0:
        # If no valid scores or all weights are zero for given rules, return a neutral score
        # Or, if scores dictionary is empty, return neutral
        if scores:
            valid_scores = [s for s in scores.values() if s is not None]
            return sum(valid_scores) / len(valid_scores) if valid_scores else 0.45
        return 0.45

    return final_score / current_total_weight # Normalize by sum of weights used


# Rule-Based Evaluation Functions (Individual and Combined)
def evaluate_statement_rules(statement_text: str, active_rules: list) -> dict:
    """
    Applies selected rule-based methods to a single statement.
    Returns a dictionary of individual scores for the active rules.
    """
    individual_scores = {}
    
    if "lexical_objectivity" in active_rules:
        try:
            individual_scores['lexical_objectivity'] = get_lexical_objectivity_score(statement_text, SUBJECTIVE_WORDS)
        except Exception:
            individual_scores['lexical_objectivity'] = 0.5

    if "hedging_modality" in active_rules:
        try:
            individual_scores['hedging_modality'] = get_hedging_modality_score(statement_text, HEDGING_WORDS)
        except Exception:
            individual_scores['hedging_modality'] = 0.5

    if "sentiment_emotion" in active_rules:
        try:
            individual_scores['sentiment_emotion'] = get_sentiment_emotion_score(statement_text)
        except Exception:
            individual_scores['sentiment_emotion'] = 0.5

    if "factual_density" in active_rules:
        try:
            individual_scores['factual_density'] = get_factual_density_score(statement_text)
        except Exception:
            individual_scores['factual_density'] = 0.1

    if "readability_style" in active_rules:
        try:
            individual_scores['readability_style'] = get_readability_style_score(statement_text)
        except Exception:
            individual_scores['readability_style'] = 0.4
            
    return individual_scores

def calculate_metrics(true_labels: pd.Series, predicted_scores: pd.Series):
    """Calculates MAE, RMSE, R2, and Std Dev of Predictions."""
    mae, rmse, r2, std_dev_predictions = np.nan, np.nan, np.nan, np.nan

    valid_mask = true_labels.notna() & predicted_scores.notna()
    true_labels_valid = true_labels[valid_mask]
    predicted_scores_valid = predicted_scores[valid_mask]

    if not true_labels_valid.empty:
        mae = mean_absolute_error(true_labels_valid, predicted_scores_valid)
        rmse = np.sqrt(mean_squared_error(true_labels_valid, predicted_scores_valid))
        std_dev_predictions = np.std(predicted_scores_valid)

        if len(true_labels_valid.unique()) > 1:
            try:
                r2 = r2_score(true_labels_valid, predicted_scores_valid)
            except ValueError:
                r2 = np.nan
        else:
            r2 = np.nan # R2 not applicable if true labels are constant
    
    return mae, rmse, r2, std_dev_predictions, len(true_labels_valid)

# Grid Search for Rule Combinations and Weights
output_root_dir = "results-rule"
os.makedirs(output_root_dir, exist_ok=True)

all_rule_names = list(RULE_WEIGHTS.keys())
overall_results_summary = []

# Ensure 'labels' are numeric for calculations
df_eval['labels'] = pd.to_numeric(df_eval['labels'], errors='coerce')

# Pre-calculate individual rule scores once to avoid redundant computation
print("Pre-calculating individual rule scores...")
for index, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc="Pre-calculating scores"):
    statement_text = str(row['statement'])
    for rule_name in all_rule_names:
        score_col = f'score_{rule_name}'
        # Check if column exists, if not, create it
        if score_col not in df_eval.columns:
            df_eval[score_col] = np.nan
        
        score_val = None
        try:
            if rule_name == "lexical_objectivity":
                score_val = get_lexical_objectivity_score(statement_text, SUBJECTIVE_WORDS)
            elif rule_name == "hedging_modality":
                score_val = get_hedging_modality_score(statement_text, HEDGING_WORDS)
            elif rule_name == "sentiment_emotion":
                score_val = get_sentiment_emotion_score(statement_text)
            elif rule_name == "factual_density":
                score_val = get_factual_density_score(statement_text)
            elif rule_name == "readability_style":
                score_val = get_readability_style_score(statement_text)
        except Exception as e:
            print(f"Error pre-calculating {rule_name} for row {index}: {e}")
            score_val = np.nan # Mark as NaN on error

        df_eval.loc[index, score_col] = score_val

print("Individual rule scores pre-calculated.")

# --- Evaluate and save single rule-based methods ---
print("\n--- Evaluating Single Rule-Based Methods ---")
single_rule_output_dir = os.path.join(output_root_dir, "single_rule")
os.makedirs(single_rule_output_dir, exist_ok=True)

# List to store metrics for single rules
single_rule_metrics_summary_list = []

for rule_name in all_rule_names:
    print(f"Evaluating single rule: {rule_name}")
    
    temp_df = df_eval.copy()
    temp_df['combined_score_single_rule'] = temp_df[f'score_{rule_name}'] # Use the pre-calculated score

    mae, rmse, r2, std_dev, valid_samples_count = calculate_metrics(
        temp_df['labels'], temp_df['combined_score_single_rule']
    )

    metrics_data = {
        "combination": rule_name,
        "weights": {rule_name: 1.0},
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Std Dev of Predictions": std_dev,
        "Valid Samples": valid_samples_count
    }
    overall_results_summary.append(metrics_data)

    # Add single rule metrics to the new list
    single_rule_metrics_summary_list.append({
        "Rule": rule_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Std Dev of Predictions": std_dev
    })

    # Save detailed CSV for the single rule
    single_rule_csv_path = os.path.join(single_rule_output_dir, f"{RULE_ABBREVIATIONS[rule_name]}_single.csv")
    temp_df[['statement', 'labels', f'score_{rule_name}', 'combined_score_single_rule']].to_csv(single_rule_csv_path, index=False)
    print(f"Saved single rule details for {rule_name} to {single_rule_csv_path}")

    # Save histogram for the single rule
    hist_filename = os.path.join(single_rule_output_dir, f'hist_{RULE_ABBREVIATIONS[rule_name]}_single.png')
    plt.figure(figsize=(10, 6))
    sns.histplot(temp_df['combined_score_single_rule'].dropna(), bins=20, kde=True, color='blue')
    plt.title(f'Histogram of Scores for Single Rule: {rule_name}')
    plt.xlabel('Reliability Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(hist_filename)
    plt.close()
    print(f"Saved histogram for single rule {rule_name} to {hist_filename}")

# Save the summary of metrics for single rule-based methods
single_rule_metrics_df = pd.DataFrame(single_rule_metrics_summary_list)
single_rule_metrics_summary_path = os.path.join(single_rule_output_dir, "single_rule_metrics_summary.csv")
single_rule_metrics_df.to_csv(single_rule_metrics_summary_path, index=False)
print(f"\nSummary of single rule metrics saved to {single_rule_metrics_summary_path}")


# Store best combinations for 2, 3, 4, 5 rules
best_combinations_by_num_rules = {} # Stores {num_rules: [{metrics, df, weights}, ...]}

# Iterate through combinations of 2 to 5 rules
for num_rules_to_combine in range(2, len(all_rule_names) + 1):
    print(f"\n--- Running Grid Search for Combinations of {num_rules_to_combine} Rules ---")
    
    current_num_rule_best_combos = [] # To store best 2 for current num_rules

    # Generate all possible combinations of rule names
    for combo in combinations(all_rule_names, num_rules_to_combine):
        combo_abbreviations = sorted([RULE_ABBREVIATIONS[rule] for rule in combo])
        combination_name_abbrev = "_".join(combo_abbreviations)
        
        # Generate weights using a grid search approach
        def generate_weights_recursive(current_rules_idx, current_weights_sum, current_weights_list):
            if current_rules_idx == len(combo) - 1:
                last_weight = round(1.0 - current_weights_sum, 1)
                # Ensure all weights are > 0.0 for the best combination consideration
                if last_weight > 0.0 and all(w > 0.0 for w in current_weights_list + [last_weight]):
                    yield current_weights_list + [last_weight]
                return

            for w in WEIGHT_STEPS:
                # Ensure individual weights are > 0.0 for rules in the combination
                if w == 0.0: # Skip 0.0 weights within the combination
                    continue
                if current_weights_sum + w <= 1.0:
                    yield from generate_weights_recursive(
                        current_rules_idx + 1,
                        current_weights_sum + w,
                        current_weights_list + [w]
                    )

        weight_combinations_raw = list(generate_weights_recursive(0, 0.0, []))
        
        # Filter for combinations where sum is 1.0 and ALL weights are strictly > 0.0
        valid_weight_combinations = []
        for wc in weight_combinations_raw:
            if np.isclose(sum(wc), 1.0) and all(w > 0.0 for w in wc):
                valid_weight_combinations.append(wc)

        if not valid_weight_combinations:
            print(f"No valid weight combinations (all weights > 0.0) found for {combination_name_abbrev}. Skipping.")
            continue

        print(f"Testing {len(valid_weight_combinations)} weight configurations for {combination_name_abbrev} (all weights > 0.0).")
        
        best_r2_for_this_combo_type = -float('inf') # Initialize with negative infinity for R2
        best_metrics_for_this_combo_type = None
        best_df_for_this_combo_type = None
        best_weights_for_this_combo_type = None

        for weights_list in tqdm(valid_weight_combinations, desc=f"Weights for {combination_name_abbrev}"):
            current_weights = {combo[i]: weights_list[i] for i in range(len(combo))}
            
            temp_df = df_eval.copy()
            temp_df[f'combined_score_{combination_name_abbrev}'] = 0.0

            # Calculate the combined score for each row using the current weights
            for index, row in temp_df.iterrows():
                individual_scores_for_combo = {rule: row[f'score_{rule}'] for rule in combo}
                combined_score = combine_scores(individual_scores_for_combo, current_weights)
                temp_df.loc[index, f'combined_score_{combination_name_abbrev}'] = combined_score
            
            # Calculate metrics for the current weight combination
            mae, rmse, r2, std_dev, valid_samples_count = calculate_metrics(
                temp_df['labels'], temp_df[f'combined_score_{combination_name_abbrev}']
            )

            # Compare by R2 score (higher is better)
            if not np.isnan(r2) and r2 > best_r2_for_this_combo_type:
                best_r2_for_this_combo_type = r2
                best_metrics_for_this_combo_type = {
                    "combination": combination_name_abbrev,
                    "weights": current_weights,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "Std Dev of Predictions": std_dev,
                    "Valid Samples": valid_samples_count
                }
                best_df_for_this_combo_type = temp_df.copy()
                best_weights_for_this_combo_type = current_weights

        if best_metrics_for_this_combo_type:
            current_num_rule_best_combos.append({
                "metrics": best_metrics_for_this_combo_type,
                "df": best_df_for_this_combo_type,
                "weights": best_weights_for_this_combo_type
            })
    
    # Sort by R2 score (descending)
    current_num_rule_best_combos.sort(key=lambda x: x['metrics']['R2'] if not np.isnan(x['metrics']['R2']) else -float('inf'), reverse=True)
    best_combinations_by_num_rules[num_rules_to_combine] = current_num_rule_best_combos[:2]


# --- Save Best Combinations (2, 3, 4, 5 rules) ---
print("\n--- Saving Best Combinations (2, 3, 4, 5 rules) ---")
for num_rules, combos_list in best_combinations_by_num_rules.items():
    for i, combo_data in enumerate(combos_list):
        metrics = combo_data['metrics']
        df_to_save = combo_data['df']
        weights = combo_data['weights']
        
        combination_name_abbrev = metrics['combination']
        
        combo_output_dir = os.path.join(output_root_dir, f"{num_rules}_rules_best")
        os.makedirs(combo_output_dir, exist_ok=True)

        # Save metrics to CSV
        metrics_filename = os.path.join(combo_output_dir, f"{combination_name_abbrev}_best_{i+1}_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_filename, index=False)
        print(f"Saved best {i+1} metrics for {num_rules} rules ({combination_name_abbrev}) to {metrics_filename}")

        # Save detailed CSV with scores
        detailed_csv_filename = os.path.join(combo_output_dir, f"{combination_name_abbrev}_best_{i+1}_detailed.csv")
        # Ensure all score columns for the specific combination are included
        score_cols_to_include = [f'score_{rule}' for rule in weights.keys()]
        df_to_save[['statement', 'labels', f'combined_score_{combination_name_abbrev}'] + score_cols_to_include].to_csv(detailed_csv_filename, index=False)
        print(f"Saved best {i+1} detailed scores for {num_rules} rules ({combination_name_abbrev}) to {detailed_csv_filename}")

        # Save histogram
        hist_filename = os.path.join(combo_output_dir, f'hist_{combination_name_abbrev}_best_{i+1}.png')
        plt.figure(figsize=(10, 6))
        sns.histplot(df_to_save[f'combined_score_{combination_name_abbrev}'].dropna(), bins=20, kde=True, color='purple')
        plt.title(f'Histogram of Combined Scores for {combination_name_abbrev} (Best {i+1} of {num_rules} Rules)')
        plt.xlabel('Combined Reliability Score')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(hist_filename)
        plt.close()
        print(f"Saved histogram for best {i+1} combination of {num_rules} rules to {hist_filename}")

# --- Overall Combinations Summary ---
print("\n--- Generating Overall Combinations Summary ---")
# Collect all metrics from single rules and the selected best combinations
final_overall_summary_list = []
for item in overall_results_summary:
    final_overall_summary_list.append(item)

for num_rules, combos_list in best_combinations_by_num_rules.items():
    for combo_data in combos_list:
        final_overall_summary_list.append(combo_data['metrics'])

overall_summary_df = pd.DataFrame(final_overall_summary_list)
overall_summary_df_path = os.path.join(output_root_dir, "overall_combinations_summary.csv")
overall_summary_df.to_csv(overall_summary_df_path, index=False)
print(f"Overall summary of selected combinations and their metrics saved to {overall_summary_df_path}")

# Display best overall performing combination
if not overall_summary_df.empty:
    filtered_overall_summary_df = overall_summary_df[
        overall_summary_df['weights'].apply(lambda x: all(w > 0.0 for w in x.values()))
    ]

    if not filtered_overall_summary_df.empty:
        overall_summary_df_sorted = filtered_overall_summary_df.sort_values(by='R2', ascending=False)
        best_overall_combo = overall_summary_df_sorted.iloc[0]
        print("\n--- Best Overall Performing Combination (by R2, with all active rule weights > 0.0) ---")
        print(best_overall_combo.to_string())

        best_overall_combo_path = os.path.join(output_root_dir, "best_overall_combination.json")
        with open(best_overall_combo_path, 'w') as f:
            serializable_combo = best_overall_combo.to_dict()
            if isinstance(serializable_combo.get('weights'), dict):
                serializable_combo['weights'] = str(serializable_combo['weights'])
            json.dump(serializable_combo, f, indent=4)
        print(f"\nBest overall combination details saved to {best_overall_combo_path}")
    else:
        print("\nNo combination found where all active rule weights are greater than 0.0. Displaying overall best without this constraint:")
        overall_summary_df_sorted = overall_summary_df.sort_values(by='R2', ascending=False)
        best_overall_combo = overall_summary_df_sorted.iloc[0]
        print(best_overall_combo.to_string())