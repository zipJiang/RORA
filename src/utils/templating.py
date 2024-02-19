"""This isolates out several
common template definitions for 
all datasets.
"""


__TEMPLATES__ = {
    "g": "{gold_rationale}",
    "s": "{base_rationale}",
    "l": "{leaky_rationale}",
    "gs": "{gold_rationale} {base_rationale}",
    "ls": "{leaky_rationale} {base_rationale}",
    "gl": "{gold_rationale} {leaky_rationale}",
    "gls": "{gold_rationale} {leaky_rationale} {base_rationale}",
    "ss": "{base_rationale} {base_rationale}",
    "n": "",
    "x": "{option_rationale}",
    "xs": "{option_rationale} {base_rationale}",
    "gpt4": "{gpt4_rationale}",
    "gpt3": "{gpt3_rationale}",
    "gpt2": "{gpt2_rationale}",
    "llama2": "{llama2_rationale}",
    "t5": "{t5_rationale}",
    "flan": "{flan_rationale}",
    "cose": "{cose_rationale}",
    "coses": "{cose_rationale} {base_rationale}",
}