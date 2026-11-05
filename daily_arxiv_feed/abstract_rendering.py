"""Typeset author abstracts without treating arbitrary source TeX as executable."""
import re

from daily_arxiv_feed.compile import _latex_escape

_MATH = re.compile(r"\$\$.*?\$\$|(?<!\\)\$(?:\\.|[^$])*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
_COMMAND = re.compile(r"\\([A-Za-z]+)")
_ALLOWED = set("""alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta
iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi
varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
frac dfrac tfrac sqrt mathrm mathbf mathit mathsf mathtt mathbb mathcal mathscr
boldsymbol text operatorname left right big Big bigg Bigg overline underline
hat widehat tilde widetilde bar vec dot ddot dagger ddots cdots ldots vdots
otimes oplus times ltimes rtimes cdot pm mp le leq ge geq neq approx sim simeq equiv propto
in notin subset subseteq supset supseteq cup cap emptyset forall exists neg land
lor to rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow
mapsto langle rangle lvert rvert lVert rVert vert Vert ket bra braket Ket Bra
Braket hbar ell partial nabla sum prod int iint oint lim min max inf sup log
ln exp sin cos tan det tr rank diag lceil rceil lfloor rfloor quad qquad
infty perp parallel prec preceq succ succeq circ bullet ast star mathop limits
nolimits underbrace overbrace stackrel binom mathrel mathord not
""".split())

_UNICODE_MATH = dict(zip(
    "αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ",
    ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa",
     "lambda","mu","nu","xi","pi","rho","sigma","tau","upsilon","phi","chi","psi","omega",
     "Gamma","Delta","Theta","Lambda","Xi","Pi","Sigma","Upsilon","Phi","Psi","Omega"],
    strict=True))
_UNICODE_MATH = {k:"\\"+v for k,v in _UNICODE_MATH.items()}
_UNICODE_MATH.update({"≤":r"\leq","≥":r"\geq","≠":r"\neq","≈":r"\approx","∞":r"\infty",
                      "±":r"\pm","×":r"\times","⊗":r"\otimes","⊕":r"\oplus","ℏ":r"\hbar",
                      "→":r"\to","←":r"\leftarrow","↔":r"\leftrightarrow","∈":r"\in",
                      "⟨":r"\langle","⟩":r"\rangle","ℝ":r"\mathbb{R}","ℂ":r"\mathbb{C}"})
_SCIENCE = re.compile("["+"".join(_UNICODE_MATH)+"]")


def _text(text: str) -> str:
    parts, start = [], 0
    for match in _SCIENCE.finditer(text):
        parts.append(_latex_escape(text[start:match.start()]))
        parts.append(r"\ensuremath{"+_UNICODE_MATH[match.group()]+"}")
        start = match.end()
    parts.append(_latex_escape(text[start:]))
    return "".join(parts)


def _safe_math(text: str) -> bool:
    if any(command not in _ALLOWED for command in _COMMAND.findall(text)):
        return False
    if re.search(r"(?<!\\)[%#]",text):
        return False
    depth = 0
    for token in re.findall(r"\\.|[{}]",text):
        if token=="{": depth += 1
        elif token=="}": depth -= 1
        if depth<0: return False
    return depth==0


def latex_abstract(abstract: str) -> str:
    result, start = [], 0
    for match in _MATH.finditer(abstract):
        result.append(_text(abstract[start:match.start()]))
        math = match.group()
        result.append(math if _safe_math(math) else _text(math))
        start = match.end()
    result.append(_text(abstract[start:]))
    return "".join(result)
