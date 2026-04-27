"""
generate_dataset.py — Creates synthetic resume dataset with bias labels
Run once: python generate_dataset.py
"""

import pandas as pd
import random
import re

random.seed(42)

# ── Bias word pools ────────────────────────────────────────────────────────────
MALE_NAMES   = ["James","John","Robert","Michael","David","William","Richard",
                 "Joseph","Thomas","Charles","Daniel","Matthew","Andrew","Joshua"]
FEMALE_NAMES = ["Mary","Patricia","Jennifer","Linda","Barbara","Elizabeth",
                 "Susan","Jessica","Sarah","Karen","Lisa","Nancy","Betty","Dorothy"]
NEUTRAL_NAMES= ["Alex","Jordan","Taylor","Morgan","Casey","Riley","Avery","Quinn"]

MALE_PRONOUNS   = ["he","him","his","himself"]
FEMALE_PRONOUNS = ["she","her","hers","herself"]

AGE_OLD_WORDS = ["experienced","veteran","seasoned","over 40","mature","senior professional",
                  "decades of experience","long career","established career","retired from"]
AGE_YOUNG_WORDS=["recent graduate","fresh graduate","young","entry-level","new to the field",
                  "just graduated","millennial","gen z","young professional"]

SKILLS = ["Python","Java","SQL","machine learning","data analysis","project management",
          "communication","teamwork","leadership","problem solving","cloud computing",
          "React","Node.js","DevOps","Agile","Scrum","TensorFlow","Docker","Kubernetes"]

COMPANIES = ["Google","Microsoft","Amazon","Infosys","TCS","Wipro","Accenture","IBM",
             "startups","mid-size company","government sector","NGO"]

DEGREES = ["B.Tech in CSE","MBA","B.Sc in Statistics","M.Tech in AI",
           "BCA","MCA","B.E. in Electronics","B.Com"]

# ── Template builder ──────────────────────────────────────────────────────────
def make_resume(name, use_pronoun, pronoun_set, age_phrase, extra_bias_words):
    skills_sample = random.sample(SKILLS, random.randint(3, 6))
    company = random.choice(COMPANIES)
    degree  = random.choice(DEGREES)
    years   = random.randint(1, 15)

    pronoun_sentence = ""
    if use_pronoun and pronoun_set:
        p = random.choice(pronoun_set)
        pronoun_sentence = f"{p.capitalize()} is a dedicated professional. "

    age_sentence = f"{age_phrase}. " if age_phrase else ""

    resume = (
        f"Name: {name}\n"
        f"{pronoun_sentence}"
        f"{age_sentence}"
        f"Education: {degree} from a reputed university.\n"
        f"Experience: {years} years at {company}.\n"
        f"Skills: {', '.join(skills_sample)}.\n"
        f"Worked on various projects involving {random.choice(SKILLS)} and {random.choice(SKILLS)}.\n"
        f"{' '.join(extra_bias_words)}"
    )
    return resume.strip()


def label_bias(name, pronoun_set, age_phrase, extra_bias_words):
    """Return (has_gender_bias, has_age_bias, has_name_bias, overall_biased)"""
    gender_bias = pronoun_set is not None
    age_bias    = age_phrase != ""
    name_bias   = name in MALE_NAMES or name in FEMALE_NAMES
    overall     = int(gender_bias or age_bias or name_bias or bool(extra_bias_words))
    return int(gender_bias), int(age_bias), int(name_bias), overall


# ── Generate rows ─────────────────────────────────────────────────────────────
rows = []
for _ in range(600):
    bias_type = random.choice(["none","gender","age","name","mixed"])

    if bias_type == "none":
        name         = random.choice(NEUTRAL_NAMES)
        pronoun_set  = None
        age_phrase   = ""
        extra        = []
    elif bias_type == "gender":
        name         = random.choice(NEUTRAL_NAMES)
        pronoun_set  = random.choice([MALE_PRONOUNS, FEMALE_PRONOUNS])
        age_phrase   = ""
        extra        = []
    elif bias_type == "age":
        name         = random.choice(NEUTRAL_NAMES)
        pronoun_set  = None
        age_phrase   = random.choice(AGE_OLD_WORDS + AGE_YOUNG_WORDS)
        extra        = []
    elif bias_type == "name":
        name         = random.choice(MALE_NAMES + FEMALE_NAMES)
        pronoun_set  = None
        age_phrase   = ""
        extra        = []
    else:  # mixed
        name         = random.choice(MALE_NAMES + FEMALE_NAMES)
        pronoun_set  = random.choice([MALE_PRONOUNS, FEMALE_PRONOUNS])
        age_phrase   = random.choice(AGE_OLD_WORDS + AGE_YOUNG_WORDS)
        extra        = random.sample(MALE_PRONOUNS + FEMALE_PRONOUNS, 2)

    use_pronoun = pronoun_set is not None
    text = make_resume(name, use_pronoun, pronoun_set, age_phrase, extra)
    gb, ab, nb, overall = label_bias(name, pronoun_set, age_phrase, extra)

    rows.append({
        "resume_text"  : text,
        "name"         : name,
        "gender_bias"  : gb,
        "age_bias"     : ab,
        "name_bias"    : nb,
        "biased"       : overall
    })

df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)
print(f"✅ dataset.csv saved — {len(df)} rows, {df['biased'].sum()} biased")
